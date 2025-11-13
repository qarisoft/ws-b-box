import torch
import torch.nn as nn
import torch.nn.functional as F

class SequentialBoostingEnsemble(nn.Module):
    def __init__(self, ResidualUNet, UNet, DeepLabV3):
        super().__init__()
        self.ResUNetmodel   = ResidualUNet
        self.UNetmodel      = UNet
        self.DeepLabV3model = DeepLabV3

        # 1) استنباط عدد قنوات الإدخال من الـ ResidualUNet
        #    (نموذجك يُخزن self.in_dim عند البناء)
        self.in_ch = getattr(ResidualUNet, 'in_dim',
                      next(m for m in ResidualUNet.modules() if isinstance(m, nn.Conv2d)).in_channels)

        # 2) طبقات تصحيح الخطأ تنتقل من (in_ch + 1) إلى in_ch
        self.error_correction1 = nn.Conv2d(self.in_ch + 1, self.in_ch, kernel_size=1)
        self.error_correction2 = nn.Conv2d(self.in_ch + 1, self.in_ch, kernel_size=1)

        # 3) أوزان الجمع النهائي
        self.weights   = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.softmax_w = nn.Softmax(dim=0)

        # طبقة ختامية للمسار النهائي
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        # إذا أرسل لك RGB أو أكثر، قصّ إلى in_ch
        if x.shape[1] != self.in_ch:
            x = x[:, :self.in_ch, :, :]

        def get_probs(logits):
            return F.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)

        # === نموذج 1 ===
        out1  = self.ResUNetmodel(x)
        prob1 = get_probs(out1)
        pred1 = prob1.argmax(dim=1, keepdim=True)
        one_hot1 = F.one_hot(pred1.squeeze(1), num_classes=prob1.shape[1])\
                      .permute(0,3,1,2).float()
        err1 = torch.abs(prob1 - one_hot1).mean(dim=1, keepdim=True)

        inp2 = self.error_correction1(torch.cat([x, err1], dim=1))

        # === نموذج 2 ===
        out2  = self.UNetmodel(inp2)
        prob2 = get_probs(out2)
        pred2 = prob2.argmax(dim=1, keepdim=True)
        one_hot2 = F.one_hot(pred2.squeeze(1), num_classes=prob2.shape[1])\
                      .permute(0,3,1,2).float()
        err2 = torch.abs(prob2 - one_hot2).mean(dim=1, keepdim=True)

        inp3 = self.error_correction2(torch.cat([x, err2], dim=1))

        # === نموذج 3 ===
        out3  = self.ResUNetmodel(inp3)
        prob3 = get_probs(out3)

        # === الجمع النهائي بالأوزان المتعلّمة ===
        w        = self.softmax_w(self.weights)
        combined = w[0]*prob1 + w[1]*prob2 + w[2]*prob3

        final_pred = combined.argmax(dim=1, keepdim=True).float()
        out        = self.final_conv(final_pred)

        return [out]
