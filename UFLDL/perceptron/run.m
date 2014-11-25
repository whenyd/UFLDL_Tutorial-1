
clear all;
load dataset3.mat

w = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)