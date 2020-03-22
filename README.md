
#环境

<br>
python3.5
tensorflow 1.4
<br>
#数据格式
<br>
联	B-PRO<br>
通	I-PRO<br>
卡	E-PRO<br>
在	O<br>
手	O<br>
机	O<br>
里	O<br>
怎	O<br>
么	O<br>
没	O<br>
有	O<br>
网	O<br>
络	O<br>
<br>
联	B-PRO<br>
通	I-PRO<br>
卡	E-PRO<br>
在	O<br>
手	O<br>
机	O<br>
里	O<br>
怎	O<br>
么	O<br>
没	O<br>
有	O<br>
网	O<br>
络	O<br>
<br>
<br>

python3 bert_crf.py --task_name=ner  --do_train=true  --vocab_file=../chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=../chinese_L-12_H-768_A-12/bert_model.ckpt --output_dir=output_crf
