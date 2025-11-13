# using bagging, make votting model that combines the three models SimpleUNet, SimpleResNet, and SimpleDeeblapV3.
# make sure to give me the best results, and please ask me if you need the dataloader class
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple model definitions for fallback


class SimpleUNet(nn.Module):
    """Simple UNet implementation for fallback"""

    def __init__(self, in_dim, out_dim, softmax, channels_in=32, dropout_rate=0.1):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_dim, channels_in, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_in, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in * 2, channels_in * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in * 4, channels_in * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(channels_in * 4, channels_in * 2, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(channels_in * 4, channels_in * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in * 2, channels_in * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(channels_in * 2, channels_in, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_in, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(channels_in, out_dim, 1)
        self.softmax = softmax

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bridge(self.pool2(e2))

        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        out = self.final(d1)
        if self.softmax:
            out = F.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)
        return [out]


class SimpleResNet(nn.Module):
    """Very simple segmentation network that guarantees exact output size matching"""

    def __init__(self, in_dim, out_dim, softmax, base_channels=32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Encoder with minimal downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4
        )

        # Decoder with exact upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, 4, stride=2, padding=1
            ),  # 1/2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels, base_channels, 4, stride=2, padding=1
            ),  # Original size
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(base_channels, out_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        original_size = x.shape[2:]

        # Encode
        x = self.encoder(x)

        # Decode
        x = self.decoder(x)

        # Final convolution
        x = self.final(x)

        # Double-check size match
        if x.shape[2:] != original_size:
            x = F.interpolate(
                x, size=original_size, mode="bilinear", align_corners=True
            )

        if self.softmax:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)

        return [x]


class SimpleDeepLabV3(nn.Module):
    """Simple DeepLabV3 implementation for fallback"""

    def __init__(self, in_dim, out_dim, softmax, channels_in=64, pretrained=True):
        super().__init__()
        # Simplified version
        self.backbone = nn.Sequential(
            nn.Conv2d(in_dim, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Simple ASPP-like module
        self.aspp1 = nn.Conv2d(256, channels_in, 1)
        self.aspp2 = nn.Conv2d(256, channels_in, 3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(256, channels_in, 3, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(256, channels_in, 3, padding=18, dilation=18)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels_in * 4, channels_in, 1), nn.ReLU(inplace=True)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_in // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                channels_in // 2, channels_in // 4, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(channels_in // 4, out_dim, 1)
        self.softmax = softmax

    def forward(self, x):
        x = self.backbone(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fusion(x)
        x = self.upsample(x)
        x = self.final(x)

        if self.softmax:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return [x]


class QuickResNet(nn.Module):

    def __init__(self, in_dim, out_dim, softmax):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Very simple architecture with no size changes
        self.network = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, 1),
        )

    def forward(self, x):
        x = self.network(x)

        if self.softmax:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)

        return [x]


class SimpleResNet2(nn.Module):
    """Fixed Simple ResNet-based segmentation with proper output sizing"""

    def __init__(self, in_dim, out_dim, softmax, backbone="resnet34", pretrained=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Initial convolution with proper padding
        self.conv1 = nn.Conv2d(in_dim, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Simple residual blocks with proper downsampling
        self.layer1 = self._make_layer(64, 64, 2)  # 1/4 size
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 1/8 size
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 1/16 size
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 1/32 size

        # Proper upsampling to restore original size
        self.upsample = nn.Sequential(
            # 1/32 -> 1/16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 1/16 -> 1/8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 1/8 -> 1/4
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 1/4 -> 1/2
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 1/2 -> Original size
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, out_dim, 1)

        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # First block with potential downsampling
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Additional blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store original size for final interpolation
        original_size = x.shape[2:]

        # Encoder path
        x = self.conv1(x)  # 1/2 size
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4 size

        x = self.layer1(x)  # 1/4 size
        x = self.layer2(x)  # 1/8 size
        x = self.layer3(x)  # 1/16 size
        x = self.layer4(x)  # 1/32 size

        # Decoder path
        x = self.upsample(x)  # Back to original size

        # Final convolution
        x = self.final(x)

        # Ensure output matches input spatial dimensions exactly
        if x.shape[2:] != original_size:
            x = F.interpolate(
                x, size=original_size, mode="bilinear", align_corners=True
            )

        # Apply final activation
        if self.softmax:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)

        return [x]


# Test function to verify output sizes
def test_model_output_sizes():
    """Test that models output correct sizes"""
    print("Testing model output sizes...")

    # Test input
    input_tensor = torch.randn(2, 1, 256, 256)
    print(f"Input shape: {input_tensor.shape}")

    models = {
        "FixedSimpleResNet": SimpleResNet2(1, 1, False),
        "SimpleResNet": SimpleResNet(1, 1, False),
    }

    for name, model in models.items():
        try:
            output = model(input_tensor)[0]
            input_size = input_tensor.shape[2:]
            output_size = output.shape[2:]

            print(f"\n{name}:")
            print(f"  Input size: {input_size}")
            print(f"  Output size: {output_size}")
            print(f"  Match: {input_size == output_size}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

            if input_size != output_size:
                print(f"  ⚠️  WARNING: Size mismatch!")
            else:
                print(f"  ✅ Size match correct!")

        except Exception as e:
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    test_model_output_sizes()

#     # Quick verification
#     print("\n" + "="*50)
#     print("QUICK FIX VERIFICATION")
#     print("="*50)

#     quick_model = QuickFixResNet(1, 1, False)
#     test_input = torch.randn(2, 1, 256, 256)
#     output = quick_model(test_input)[0]

#     print(f"Input shape: {test_input.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Size match: {test_input.shape == output.shape}")
#     print(f"Parameters: {sum(p.numel() for p in quick_model.parameters()):,}")
