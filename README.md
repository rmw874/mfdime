# mfdime

DiME explains an image classifier with counterfactuals: it edits a photo step by step until the classifier changes its answer. At every step it looks ahead at the finished image by walking the generator through all the remaining steps (with fresh noise at each; the noise-free walk takes the same steps without it), takes the classifier's gradient on that look-ahead and applies it straight to the noisy image (DiME's shortcut) instead of differentiating through the look-ahead (the exact gradient). We replace the walk with one jump of a flow map: DMF on ImageNet and FiRe's pixel MeanFlow on CelebA-HQ faces.

## Hypotheses

### H1 
With DiME's shortcut, one jump gives a near-identical flip rate and sFID (ACE) as the noise-free walk over 5 seeds, at 3055/212 approx= 14x fewer calls to the generator.

### H2
With the same DiME settings and flow-map look-ahead, full backpropagation through the look-ahead improves target-class
success on a classifier that never guided the edits, compared with DiME's shortcut gradient.

The comparison uses the same gains and sampling schedule, not equal realised update norms: gradient magnitude and
direction are both allowed to change. Independent-classifier success measures transfer, a proxy for semantic validity,
not a direct measurement of visual realism.

### H3 
The shortcut points away from the exact gradient most at high noise and carries more high-frequency energy. The jump's exact gradient points the same way as the walk's.
