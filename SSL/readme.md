- **Self-Supervised Pre-Training**
```bash
python /hdchoi00/SSL/moco/moco-v3/main_moco.py \
  -a vit_base \
  --resume /hdchoi00/SSL/moco/weights/vit-b-300ep.pth.tar \
  --batch-size 512 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=600 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.1 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  /hdchoi00/SSL/moco/data
```
