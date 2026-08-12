#ifndef ML_FIXED_VGG_H
#define ML_FIXED_VGG_H

#include "volefier/ml-fixed/fixed_resnet.h" // reuses the ResNet/CNN builder ops
#include <string>
using namespace emp;

// VGG-11/13/16/19 (optionally with BatchNorm). VGG is a strict subset of the
// ResNet ops -- conv(3x3) -> [BN] -> ReLU stacks, a maxpool per stage, then
// three FC layers -- with NO residuals/bottleneck. Built directly on the
// ResNet builder's ops; no new gadgets.

struct VGGCfg {
  int convs[5]; // conv layers per stage
  int chans[5]; // channels per stage
  int fc_dim, classes, in_ch, in_hw;
  bool bn;
  std::string name;
};

static inline VGGCfg vgg_cfg(int depth, int base, int fc_dim, int classes, int in_hw,
                             bool bn = false) {
  VGGCfg c;
  c.fc_dim = fc_dim;
  c.classes = classes;
  c.in_ch = 3;
  c.in_hw = in_hw;
  c.bn = bn;
  c.chans[0] = base;
  c.chans[1] = base * 2;
  c.chans[2] = base * 4;
  c.chans[3] = base * 8;
  c.chans[4] = base * 8;
  const char *nm;
  int t[5];
  switch (depth) {
  case 11: {
    int a[5] = {1, 1, 2, 2, 2};
    for (int i = 0; i < 5; i++) t[i] = a[i];
    nm = "vgg11";
  } break;
  case 13: {
    int a[5] = {2, 2, 2, 2, 2};
    for (int i = 0; i < 5; i++) t[i] = a[i];
    nm = "vgg13";
  } break;
  case 16: {
    int a[5] = {2, 2, 3, 3, 3};
    for (int i = 0; i < 5; i++) t[i] = a[i];
    nm = "vgg16";
  } break;
  default: {
    int a[5] = {2, 2, 4, 4, 4};
    for (int i = 0; i < 5; i++) t[i] = a[i];
    nm = "vgg19";
  } break;
  }
  for (int i = 0; i < 5; ++i) c.convs[i] = t[i];
  c.name = (bn ? std::string(nm) + "_bn" : std::string(nm));
  return c;
}

// conv-stack stages -> flatten -> FC(relu) -> FC(relu) -> FC(logits)
static inline VecSt vgg_forward(ResNet &net, const VGGCfg &cfg, const St &input) {
  St x = input;
  for (int s = 0; s < 5; ++s) {
    for (int i = 0; i < cfg.convs[s]; ++i) {
      x = net.conv(x, cfg.chans[s], 3, 1, 1);
      if (cfg.bn) x = net.bn(x);
      x = net.relu(x);
    }
    x = net.maxpool(x, 2, 2, 0);
  }
  VecSt v = net.flatten(x);
  v = net.relu_vec(net.fc(v, cfg.fc_dim));
  v = net.relu_vec(net.fc(v, cfg.fc_dim));
  return net.fc(v, cfg.classes);
}

#endif
