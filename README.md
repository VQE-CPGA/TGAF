# TGAF
The Mindspore code of "Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion".

# Requirements
CUDA==11.6 mindspore==2.2.14 python==3.7.5

# Dataset 
Training Data:[MFQEv2](https://github.com/ryanxingql/mfqev2.0)

Test Data:[MFQEv2](https://github.com/ryanxingql/mfqev2.0)

# Train and Test
```python
python train_SPL.py
python test_SPL.py
```


# Acknowledgments
Thanks for the support of Huawei MindSpore Platform for training our model. 

# Citation
```python
@article{zhu2024compressed,
  title={Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion},
  author={Qiang, Zhu and Yajun, Qiu and Yu, Liu and Shuyuan, Zhu and Bing, Zeng},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE},
  doi={10.1109/LSP.2024.3407536}
}
```
