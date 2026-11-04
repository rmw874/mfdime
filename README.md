# mfdime

DiME explains an image classifier with counterfactuals: it edits a photo step by step until the classifier changes its answer. At every step it looks ahead at the finished image by walking the generator through all the remaining steps (with fresh noise at each; the noise-free walk takes the same steps without it), takes the classifier's gradient on that look-ahead and applies it straight to the noisy image (DiME's shortcut) instead of differentiating through the look-ahead (the exact gradient). We replace the walk with one jump of a flow map: DMF on ImageNet and FiRe's pixel MeanFlow on CelebA-HQ faces.

## Hypotheses

### H1 
With DiME's shortcut, one jump gives a near-identical flip rate and sFID (ACE) as the noise-free walk over 5 seeds, at 3055/212 approx= 14x fewer calls to the generator.

### H2
At equal step size, the exact gradient gives more realistic edits, i.e. more flips of a classifier that never guided the edits.

### H3 
The shortcut points away from the exact gradient most at high noise and carries more high-frequency energy. The jump's exact gradient points the same way as the walk's.
