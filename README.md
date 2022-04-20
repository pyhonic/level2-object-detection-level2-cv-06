## 오마카세 조 Detection

## Train
```
$ python python swin_train.py \
	--base_path [model 경로] --data_dir [data 경로] \
    --config_file [config 경로] --n_fold [CV 경로] \
    -- resume_from [resume할 pth 경로]
```

## 📄 Experiments
- [Notion](https://long-knuckle-1e6.notion.site/Lv-02-P-stage01-97aa6b55968e4a02a74355fd59cabe29)

## reference
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Boxinst](https://github.com/wangbo-zhao/OpenMMLab-BoxInst)