# CycleGAN

## Experimental Results

### Horse to Zebra


<table>
  <tr>
    <td>Description</td>
    <td>Input image</td>
    <td>Result</td>
  </tr>
  <tr>
    <td>Horse to zebra</td>
    <td><img src="./results/horse2zebra/horse.png"></td>
    <td><img src="./results/horse2zebra/horse2zebra.png"></td>
  </tr>
  <tr>
    <td>Zebra to horse</td>
    <td><img src="./results/horse2zebra/zebra.png"></td>
    <td><img src="./results/horse2zebra/zebra2horse.png"></td>
  </tr>
</table>

### Cityscapes to IDD


<table>
  <tr>
    <td>Domain</td>
    <td>Input image</td>
    <td>Random crop by 256 w/o identity loss</td>
    <td>Random crop by 1024 w/o identity loss</td>
    <td>Random crop by 256 w/ identity loss</td>
  </tr>
  <tr>
    <td>Cityscapes to IDD</td>
    <td><img src="./results/German2India/A2B/input.png"></td>
    <td><img src="./results/German2India/A2B/Crop256.png"></td>
    <td><img src="./results/German2India/A2B/Crop1024.png"></td>
    <td><img src="./results/German2India/A2B/withIdentity_Crop256.png"></td>
  </tr>
  <tr>
    <td>IDD to Cityscapes</td>
    <td><img src="./results/German2India/B2A/original.png"></td>
    <td><img src="./results/German2India/B2A/Crop256.png"></td>
    <td><img src="./results/German2India/B2A/Crop1024.png"></td>
    <td><img src="./results/German2India/B2A/withIdentity_Crop256.png"></td>
  </tr>
</table>


<table>
  <tr>
    <td>Domain</td>
    <td>Input image</td>
    <td>Random crop by 1024 w/ identity loss</td>
    <td>Random crop by 1024 w/ identity loss and use 15 residual blocks</td>
    <td>Reize by 1024 w/ identity loss</td>
  </tr>
  <tr>
    <td>Cityscapes to IDD</td>
    <td><img src="./results/German2India/A2B/input.png"></td>
    <td><img src="./results/German2India/A2B/withIdentity-crop1024.png"></td>
    <td><img src="./results/German2India/A2B/ResidualBlock15_withIdentity_Crop1024.png"></td>
    <td><img src="./results/German2India/A2B/withIdentity_Resize1024.png"></td>
  </tr>
  <tr>
    <td>IDD to Cityscapes</td>
    <td><img src="./results/German2India/B2A/original.png"></td>
    <td><img src="./results/German2India/B2A/withIdentity_Crop1024.png"></td>
    <td><img src="./results/German2India/B2A/Residual15_withIdentity_Crop1024.png"></td>
    <td><img src="./results/German2India/B2A/withIdentity_Resize1024.png"></td>
  </tr>
</table>
