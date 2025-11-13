import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_3_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

        # Secondary branch
        # FIX: Use in_dim instead of hardcoded values
        self.conv0 = nn.Conv2d(
            in_dim, int(in_dim / projectionFactor), kernel_size=2, stride=2
        )
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(
            int(in_dim / projectionFactor),
            int(in_dim / projectionFactor),
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        maxpool_output, indices = self.maxpool0(input)

        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)
        do = self.do(p2)

        # Zero padding to match dimensions
        depth_to_pad = abs(maxpool_output.shape[1] - do.shape[1])
        if depth_to_pad > 0:
            padding = torch.zeros(
                maxpool_output.shape[0],
                depth_to_pad,
                maxpool_output.shape[2],
                maxpool_output.shape[3],
                device=maxpool_output.device,
            )
            maxpool_output_pad = torch.cat((maxpool_output, padding), 1)
        else:
            maxpool_output_pad = maxpool_output

        output = maxpool_output_pad + do
        final_output = self.PReLU3(output)

        return final_output, indices


class BottleNeckNormal(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_3_3(
            int(in_dim / projectionFactor), int(in_dim / projectionFactor)
        )
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim != out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim != self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do

        output = self.PReLU_out(output)
        return output


class ENet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 16):
        super().__init__()
        self.projecting_factor = 4
        self.n_kernels = channels_in

        # Initial
        self.conv0 = nn.Conv2d(in_dim, 15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

        # First group - FIXED: Proper channel dimensions
        self.bottleNeck1_0 = BottleNeckDownSampling(
            16,
            self.projecting_factor,
            self.n_kernels * 4,  # Input: 16 channels (15+1 from concat)
        )
        self.bottleNeck1_1 = BottleNeckNormal(
            self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01
        )
        self.bottleNeck1_2 = BottleNeckNormal(
            self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01
        )
        self.bottleNeck1_3 = BottleNeckNormal(
            self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01
        )
        self.bottleNeck1_4 = BottleNeckNormal(
            self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01
        )

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(
            self.n_kernels * 4, self.projecting_factor, self.n_kernels * 8
        )
        self.bottleNeck2_1 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_2 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_3 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_4 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_5 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_6 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_7 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck2_8 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_2 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_3 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_4 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_5 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_6 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_7 = BottleNeckNormal(
            self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1
        )
        self.bottleNeck3_8 = BottleNeckNormal(
            self.n_kernels * 8,
            self.n_kernels * 4,
            self.projecting_factor,
            0.1,  # Reduce channels for decoder
        )

        # ### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)
        self.bottleNeck_Up_1_0 = BottleNeckNormal(
            self.n_kernels * 8,
            self.n_kernels * 4,
            self.projecting_factor,
            0.1,  # Input: 64 (from cat), Output: 16
        )
        self.PReLU_Up_1 = nn.PReLU()

        self.bottleNeck_Up_1_1 = BottleNeckNormal(
            self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.1
        )
        self.bottleNeck_Up_1_2 = BottleNeckNormal(
            self.n_kernels * 4,
            self.n_kernels,
            self.projecting_factor,
            0.1,  # Reduce to initial channels
        )

        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        self.bottleNeck_Up_2_1 = BottleNeckNormal(
            self.n_kernels * 2,
            self.n_kernels,
            self.projecting_factor,
            0.1,  # Input: 32 (from cat), Output: 16
        )
        self.bottleNeck_Up_2_2 = BottleNeckNormal(
            self.n_kernels, self.n_kernels, self.projecting_factor, 0.1
        )
        self.PReLU_Up_2 = nn.PReLU()

        # Final upsampling
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.n_kernels, self.n_kernels, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.n_kernels),
            nn.PReLU(),
        )

        # Output layers
        self.out_025 = nn.Conv2d(
            self.n_kernels * 8, out_dim, kernel_size=3, stride=1, padding=1
        )
        self.out_05 = nn.Conv2d(
            self.n_kernels, out_dim, kernel_size=3, stride=1, padding=1
        )
        self.final = nn.Conv2d(self.n_kernels, out_dim, kernel_size=1)

        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

        self.init_weights()
        print(f"Initialized {self.__class__.__name__} successfully")

    def forward(self, input):
        # Initial block
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0, indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)  # 15 + 1 = 16 channels

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)  # 16 -> 64 channels
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)  # 64 -> 128 channels
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)

        # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)  # 128 -> 64 channels

        # #### Deconvolution Path ####
        # First block
        unpool_0 = self.unpool_0(bn3_8, indices_2)
        bn_up_1_0 = self.bottleNeck_Up_1_0(
            torch.cat((unpool_0, bn1_4), dim=1)
        )  # 64 + 64 = 128 -> 64
        up_block_1 = self.PReLU_Up_1(unpool_0 + bn_up_1_0)
        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)  # 64 -> 16 channels

        # Second block
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)
        bn_up_2_1 = self.bottleNeck_Up_2_1(
            torch.cat((unpool_1, outputInitial), dim=1)
        )  # 16 + 16 = 32 -> 16
        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)
        up_block_2 = self.PReLU_Up_2(unpool_1 + bn_up_2_2)

        # Final upsampling
        unpool_12 = self.deconv3(up_block_2)
        output = self.final(unpool_12)
        output = self.pred_func(output)

        return [output]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
