import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# 84k parameters architecture

class CustomSelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.A = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, L, D = x.shape
        projected = self.x_proj(x)
        delta, B_mat, C_mat = torch.split(projected, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(delta) 
        dA = torch.exp(torch.einsum("bld,n->bldn", delta, -torch.exp(self.A)))
        
        hidden_state = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        
        for t in range(L):
            x_t = x[:, t, :]
            delta_t = delta[:, t, 0]
            B_t = B_mat[:, t, :]
            C_t = C_mat[:, t, :]
            
            dB_t = torch.einsum("b,bn->bn", delta_t, B_t)
            dA_t = dA[:, t, 0, :]
            
            hidden_state = dA_t[:, None, :] * hidden_state + torch.einsum("bd,bn->bdn", x_t, dB_t)
            y_t = torch.einsum("bdn,bn->bd", hidden_state, C_t) + x_t * self.D
            outputs.append(y_t)
            
        return self.out_proj(torch.stack(outputs, dim=1))

class BidirectionalSpatialMamba(nn.Module):
    def __init__(self, d_model, d_state=8):
        super().__init__()
        self.fwd_ssm = CustomSelectiveScan(d_model=d_model, d_state=d_state)
        self.bwd_ssm = CustomSelectiveScan(d_model=d_model, d_state=d_state)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        out_fwd = self.fwd_ssm(x)
        x_bwd = torch.flip(x, dims=[1])
        out_bwd = self.bwd_ssm(x_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])
        return self.ln(out_fwd + out_bwd) + x

class DermoMambaEdgeNet(nn.Module):
    def __init__(self, num_classes=7, metadata_dim=3, d_model=64, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.cnn_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        in_patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(in_patch_dim, d_model)
        self.mamba_backbone = nn.Sequential(
            BidirectionalSpatialMamba(d_model=d_model, d_state=8),
            BidirectionalSpatialMamba(d_model=d_model, d_state=8)
        )
        
        self.metadata_stream = nn.Sequential(
            nn.Linear(metadata_dim, 16),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 + d_model + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def _extract_patches(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        return patches.contiguous().view(B, -1, C * self.patch_size * self.patch_size)

    def forward(self, img, meta):
        feat_cnn = self.cnn_stream(img).squeeze(-1).squeeze(-1)
        patches = self._extract_patches(img)
        tokens = self.patch_embedding(patches)
        feat_ssm = torch.mean(self.mamba_backbone(tokens), dim=1)
        feat_meta = self.metadata_stream(meta)
        
        fused = torch.cat((feat_cnn, feat_ssm, feat_meta), dim=1)
        return self.classifier(fused)

# STANDALONE INFERENCE PIPELINE WRAPPER FOR PUBLIC DEPLOYMENT

class DermoMambaInferencePipeline:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate your exact compiled model architecture
        self.model = DermoMambaEdgeNet(num_classes=7, metadata_dim=3, d_model=64, patch_size=16)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        # Clinical categories matching the HAM10000 layout
        self.classes = ['Actinic', 'BCC', 'Benign', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run_diagnosis(self, image_path, age, is_male, site_index):
        """
        Executes unified tri-stream evaluation on a local edge sample.
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Structure clinical covariates vector
        meta_tensor = torch.tensor([[float(age) / 100.0, float(is_male), float(site_index)]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor, meta_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        return {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}

if __name__ == "__main__":
    import os
    
    weights_file = "wrapper_demo.pth" if not os.path.exists("dermomamba_fusion_final.pth") else "dermomamba_fusion_final.pth"
    print(f"🔒 Initializing diagnostic pipeline engine utilizing: {weights_file}")
    
    try:
        diagnostic_engine = DermoMambaInferencePipeline(weights_path=weights_file)
        
        # Mock test inputs to verify pipeline execution loop
        print("💡 Pipeline constructed. Creating temporary edge arrays for inference verification...")
        dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        dummy_img.save("temp_inference_sample.jpg")
        
        prediction_results = diagnostic_engine.run_diagnosis(
            image_path="temp_inference_sample.jpg", 
            age=45, 
            is_male=1,      # 1: Male, 0: Female
            site_index=2    # Multi-class mapped anatomical index
        )
        
        print("\n🩺 DermoMamba-Fusion Standalone Inference Probabilities:")
        for lesion, score in prediction_results.items():
            print(f" - {lesion:15s}: {score * 100:6.2f}%")
            
        # Clean workspace
        if os.path.exists("temp_inference_sample.jpg"):
            os.remove("temp_inference_sample.jpg")
            
    except Exception as e:
        print(f"\n❌ Pipeline execution test failed. Technical error: {str(e)}")
