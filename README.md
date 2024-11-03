# InsEYEte: AI-Powered Glaucoma Detection and Analysis

InsEYEte is an advanced application designed to assist in the early detection and analysis of glaucoma using deep learning models. By processing fundus images, InsEYEte provides predictions on the presence of glaucoma and identifies specific features associated with the condition.  

## Dataset Used
This model was trained on the Justified Referral in AI Glaucoma Screening (JustRAIGS) dataset which includes over 100,000 color fundus photos (CFP) alongside glaucoma referrals and findings to justify referral from several graders. 

## Features

- **Glaucoma Detection**: Utilizes a ConvNeXt model to predict the likelihood of glaucoma from uploaded fundus images.
- **Feature Analysis**: Identifies and displays key features indicative of glaucoma, such as:
  - Appearance of the neuroretinal rim superiorly (ANRS)
  - Appearance of the neuroretinal rim inferiorly (ANRI)
  - Retinal nerve fiber layer defect superiorly (RNFLDS)
  - Retinal nerve fiber layer defect inferiorly (RNFLDI)
  - Baring circumlinear vessel superiorly (BCLVS)
  - Baring circumlinear vessel inferiorly (BCLVI)
  - Nasalization of vessel trunk (NVT)
  - Disc hemorrhages (DH)
  - Laminar dots (LD)
  - Large cup (LC)
- **Interactive Interface**: Allows users to upload images, view predictions, and adjust feature justifications through a user-friendly interface.

## License
This project is licensed under the MIT License. See the LICENSE file for details.