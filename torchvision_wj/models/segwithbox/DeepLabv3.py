import torch
import torch.nn as nn
import torchvision.models as models


class DeepLabV3(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 32):
        super(DeepLabV3, self).__init__()

        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=True)

        # Modify first conv layer for input channels
        resnet.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Use ResNet as backbone (remove avgpool and fc layers)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ASPP module
        self.aspp = ASPP(
            in_channels=2048, out_channels=256
        )  # ResNet-50 output has 2048 channels

        # Final classifier
        self.classifier = nn.Conv2d(256, out_dim, kernel_size=1)

        # Upsampling to original size
        self.upsample = nn.Upsample(
            size=(256, 256), mode="bilinear", align_corners=True
        )

        # Activation function
        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

        print(f"Initialized {self.__class__.__name__} successfully")

    def forward(self, x):
        # Get original input size for upsampling
        original_size = x.shape[2:]  # (H, W)

        # Backbone feature extraction
        x = self.backbone(x)  # Output: [batch_size, 2048, H/32, W/32]

        # Apply ASPP
        x = self.aspp(x)  # Output: [batch_size, 256, H/32, W/32]

        # Classification
        x = self.classifier(x)  # Output: [batch_size, out_dim, H/32, W/32]

        # Upsample to original size
        x = nn.functional.interpolate(
            x, size=original_size, mode="bilinear", align_corners=True
        )

        # Apply final activation
        output = self.pred_func(x)

        return [output]  # Return as list to match UNetWithBox expectations


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # Atrous convolutions with different rates
        self.atrous_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=6, dilation=6
        )
        self.atrous_3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=12, dilation=12
        )
        self.atrous_4 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=18, dilation=18
        )

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Fusion convolution
        self.conv_1x1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Store original size
        original_size = x.shape[2:]

        # Atrous convolutions
        x1 = self.atrous_1(x)
        x2 = self.atrous_2(x)
        x3 = self.atrous_3(x)
        x4 = self.atrous_4(x)

        # Global average pooling branch
        global_avg = self.global_avg_pool(x)
        global_avg = nn.functional.interpolate(
            global_avg, size=original_size, mode="bilinear", align_corners=True
        )

        # Concatenate all branches
        x = torch.cat([x1, x2, x3, x4, global_avg], dim=1)

        # Fusion
        x = self.conv_1x1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


# import torch
# import torch.nn as nn
# import torchvision.models as models


# class DeepLabV3(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 32):
#         super(DeepLabV3, self).__init__()

#         # اختيار الـ backbone (مثل ResNet-50 أو ResNet-101)
#         resnet = models.resnet50(pretrained=True)

#         # تعديل الطبقة الأولى من ResNet لاستقبال 1 قناة بدلاً من 3 قنوات
#         resnet.conv1 = nn.Conv2d(
#             in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#         )

#         # إزالة الطبقات العلوية من الـ ResNet لاستخدامها كـ backbone
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         self.upsample = nn.Upsample(
#             size=(256, 256), mode="bilinear", align_corners=True
#         )

#         # تحديد عدد القنوات التي تخرج من الـ backbone
#         backbone_out_channels = (
#             2048  # هذه القيمة تعتمد على الـ ResNet المستخدم (مثال: 2048 في ResNet-50)
#         )

#         # طبقة تقليص القنوات من 2048 إلى 128
#         self.conv_to_128_channels = nn.Conv2d(
#             backbone_out_channels, 224, kernel_size=1, stride=1, padding=0
#         )
#         self.conv_to_32_128_1x1 = nn.Conv2d(
#             224, 128, kernel_size=8, stride=8, padding=0
#         )  # لتحويل الأبعاد إلى 1x1 مع 128 قناة
#         self.conv_to_128_5 = nn.Conv2d(
#             224, 128, kernel_size=1, stride=1, padding=0
#         )  # تقليص عدد القنوات من 224 إلى 128

#         # أضف الطبقة التوسعية (Atrous Convolution) في النهاية
#         self.aspp = ASPP(in_channels=224, out_channels=channels_in)

#         # طبقة ختامية لتقليص الأبعاد إلى عدد الفئات
#         self.classifier = nn.Conv2d(channels_in, out_dim, kernel_size=1)

#         # دالة تنشيط بناءً على خيار softmax أو sigmoid
#         if softmax:
#             self.pred_func = nn.Softmax(dim=1)
#         else:
#             self.pred_func = nn.Sigmoid()
#         print(f"Initialized {self.__class__.__name__} succesfully")

#     def forward(self, x):
#         # تمرير الصور عبر الـ Backbone (ResNet)
#         x = self.backbone(x)

#         # تقليص القنوات إلى 128
#         x = self.conv_to_128_channels(x)
#         # x=self.conv_to_32_128_1x1(x)
#         x = x.view(5, 224, 8, 8)  # تحويل الأبعاد إلى [5, 128, 8, 8]

#         # تمرير البيانات عبر الطبقة التوسعية (Atrous Convolution)
#         x = self.aspp(x)
#         x = self.upsample(x)

#         # تطبيق الطبقة النهائية (الـ Classifier)
#         x = self.classifier(x)

#         # تطبيق الدالة النهائية (Softmax أو Sigmoid)
#         output = self.pred_func(x)

#         return output


# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()

#         # طبقات Atrous Convolution مع معدلات مختلفة
#         self.atrous_1 = nn.Conv2d(
#             in_channels, out_channels, kernel_size=3, padding=6, dilation=6
#         )
#         self.atrous_2 = nn.Conv2d(
#             in_channels, out_channels, kernel_size=3, padding=12, dilation=12
#         )
#         self.atrous_3 = nn.Conv2d(
#             in_channels, out_channels, kernel_size=3, padding=18, dilation=18
#         )
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # جمع المعلومات من كامل الصورة

#         # طبقة دمج
#         self.conv_1x1 = nn.Conv2d(out_channels * 10, out_channels, kernel_size=1)
#         self.batch_norm = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # الحصول على الإخراج من كل طبقة Atrous Convolution
#         x1 = self.atrous_1(x)
#         x2 = self.atrous_2(x)
#         x3 = self.atrous_3(x)

#         # تطبيق الـ Adaptive Average Pooling
#         global_avg = self.global_avg_pool(x)
#         global_avg = global_avg.expand(
#             -1, -1, x.size(2), x.size(3)
#         )  # التوسيع ليأخذ نفس حجم الصورة الأصلية

#         # دمج النتائج
#         x = torch.cat([x1, x2, x3, global_avg], dim=1)

#         # تمرير البيانات عبر طبقة الـ 1x1 Convolution والـ Batch Normalization
#         x = self.conv_1x1(x)
#         x = self.batch_norm(x)
#         x = self.relu(x)

#         return x
