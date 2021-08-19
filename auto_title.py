## 自动摘要的例子
import sys

import torch
import os
import time
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.Unilm import Unilm
from rouge import Rouge
from bert_seq2seq.utils import get_args


def seed_everything(my_seed=606):
    np.random.seed(my_seed)
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_file(article_path, summary_path):

    summary = pd.read_csv(summary_path, encoding='utf8', sep='\t', names=['summary'])
    article = pd.read_csv(article_path, encoding='utf8', sep='\t', names=['article'])

    tgt = list(summary['summary'].apply(lambda x: x.strip('\n').lower()))
    src = list(article['article'].apply(lambda x: x.strip('\n').lower() if len(x) < 512 else x[:512]))

    return src, tgt

    
class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt, word2idx) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt, max_length=512)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class Trainer:
    def __init__(self, config, word2idx):
        # 加载数据
        self.data_src, self.data_tgt = read_file(config['dataset']['article_path'], config['dataset']['summary_path'])
        self.sents_src, self.sents_tgt = self.data_src[:config['dataset']['set_train_num']], self.data_tgt[:config['dataset']['set_train_num']]

        # 判断是否有可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = Unilm(word2idx, model_name=config['model']['model_name'])
        self.bert_model.set_device(self.device)
        ## 加载预训练的模型参数～  
        self.bert_model.load_pretrain_params(config['model']['model_path'])

        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=config['solver']['lr'], weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt, word2idx)
        self.dataloader = DataLoader(dataset, batch_size=config['solver']['batch_size'], shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True, file=sys.stdout):
            step += 1
            if step % 300 == 0:
                self.bert_model.eval()
                test_data = ["本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
                 "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献", 
                 "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3))
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 8000 == 0:
                self.save(config['model']['model_save_path'])

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                )
            report_loss += loss.item()
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(config['model']['model_save_path'])


def rouge(a, b):
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    r1 = rouge_score[0]["rouge-1"]
    r2 = rouge_score[0]["rouge-2"]
    rl = rouge_score[0]["rouge-l"]

    return r1['f'], r2['f'], rl['f']


def evaluate(epoch, config):
    print('-'*50, 'start to evaluate', '-'*50)
    # 加载字典
    word2idx = load_chinese_base_vocab(config['model']['vocab_path'])
    # 定义模型
    bert_model = Unilm(word2idx, tokenizer=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=config['model']['model_save_path'], device=device)

    data_src, data_tgt = read_file(config['dataset']['article_path'], config['dataset']['summary_path'])

    # rouge-1, rouge-2, rouge-l
    rouge1, rouge2, rougel = 0, 0, 0
    data_src, data_tgt = data_src[config['dataset']['set_eval_num']:], data_tgt[config['dataset']['set_eval_num']:]
    for text, target in zip(data_src, data_tgt):
        with torch.no_grad():
            pred = list(bert_model.generate(text, beam_size=1))
        target = list(target)
        r1, r2, rl = rouge(' '.join(pred), ' '.join(target))
        rouge1 += r1
        rouge2 += r2
        rougel += rl

    print(f"epoch : {str(epoch)}     ROUGE-1: {rouge1 / 300}     ROUGE-2: {rouge2 / 300}    ROUGE-L: {rougel / 300}")


if __name__ == '__main__':

    args = get_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    seed_everything(my_seed=config['solver']['seed'])

    word2idx = load_chinese_base_vocab(config['model']['vocab_path'], simplfied=False)

    trainer = Trainer(config, word2idx)

    for epoch in range(config['solver']['epoch']):
        # 训练一个epoch
        trainer.train(epoch)
        evaluate(epoch, config)