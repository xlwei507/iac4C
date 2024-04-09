import sys
from Bio import SeqIO
import pandas as pd

def fasta_to_csv(fasta_file, csv_file):
    # 读取fasta文件
    records = list(SeqIO.parse(fasta_file, "fasta"))
    
    # 提取序列ID和序列
    data = [(record.id, str(record.seq)) for record in records]
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=["ID", "Sequence"])
    
    # 将DataFrame保存为csv文件
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    # 从命令行参数获取文件名
    fasta_file = sys.argv[1]
    csv_file = sys.argv[2]

    # 使用函数
    fasta_to_csv(fasta_file, csv_file)
