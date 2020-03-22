# Chinese-NER-
环境
====
python3.5
tensorflow 1.4
数据格式
====
联	B-PRO
通	I-PRO
卡	E-PRO
在	O
手	O
机	O
里	O
怎	O
么	O
没	O
有	O
网	O
络	O

联	B-PRO
通	I-PRO
卡	E-PRO
在	O
手	O
机	O
里	O
怎	O
么	O
没	O
有	O
网	O
络	O

python3 bert_crf.py --task_name=ner  --do_train=true  --vocab_file=../chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=../chinese_L-12_H-768_A-12/bert_model.ckpt --output_dir=output_crf
