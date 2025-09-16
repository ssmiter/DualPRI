#!/usr/bin/env python3
import os
import subprocess
import glob
import re
from collections import defaultdict

# 氨基酸和核苷酸的三字母到单字母映射
AMINO_ACIDS = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

NUCLEOTIDES = {
    'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U', 'T': 'T',
    'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'T',
    'ADE': 'A', 'GUA': 'G', 'CYT': 'C', 'THY': 'T', 'URA': 'U',
    # 更多可能的名称
    'RAD': 'A', 'RG': 'G', 'RC': 'C', 'RU': 'U'
}

# 指定PDB文件目录
PDB_DIR = "./Dataset/S394_pdbs"
OUTPUT_DIR = "./processed_pdbs"


def create_tcl_script():
    """创建VMD TCL脚本文件"""
    # 确保输出目录存在
    os.makedirs(f"{OUTPUT_DIR}/protein_chains", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/nucleic_chains", exist_ok=True)

    script_content = f"""
# 创建输出目录
file mkdir {OUTPUT_DIR}/protein_chains
file mkdir {OUTPUT_DIR}/nucleic_chains

# 处理指定目录中的所有PDB文件
foreach pdbfile [glob {PDB_DIR}/*.pdb] {{
    # 提取PDB ID (不包括路径和.pdb扩展名)
    set pdbid [file rootname [file tail $pdbfile]]
    puts "Processing $pdbid..."

    # 加载PDB文件
    mol new $pdbfile

    # 分离蛋白质和核酸链
    set protein [atomselect top "protein"]
    set nucleic [atomselect top "nucleic"]

    # 写入分离的PDB文件
    $protein writepdb {OUTPUT_DIR}/protein_chains/${{pdbid}}_protein.pdb
    $nucleic writepdb {OUTPUT_DIR}/nucleic_chains/${{pdbid}}_nucleic.pdb

    # 清除内存
    $protein delete
    $nucleic delete
    mol delete top
}}

puts "All PDB files processed."
# 退出VMD
quit
"""
    with open(f"{OUTPUT_DIR}/separate_all_chains.tcl", "w") as f:
        f.write(script_content)
    print(f"TCL脚本已创建: {OUTPUT_DIR}/separate_all_chains.tcl")


def run_vmd_script():
    """运行VMD脚本分离链"""
    try:
        print("运行VMD批处理脚本...")
        result = subprocess.run(
            ["vmd", "-dispdev", "text", "-e", f"{OUTPUT_DIR}/separate_all_chains.tcl"],
            check=True,
            text=True,
            capture_output=True
        )
        print("VMD处理完成。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"VMD执行错误: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("找不到VMD命令。请确保VMD已安装且在PATH中。")
        return False


def extract_protein_sequence(pdb_file):
    """从蛋白质PDB文件中提取氨基酸序列"""
    if not os.path.exists(pdb_file):
        print(f"文件不存在: {pdb_file}")
        return {}

    # 存储序列，按链ID和残基编号组织
    chains = defaultdict(dict)

    # PDB文件中ATOM记录的氨基酸残基
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":  # 只考虑Alpha Carbon，避免重复
                chain_id = line[21]
                res_num = int(line[22:26])
                res_name = line[17:20].strip()

                if res_name in AMINO_ACIDS:
                    chains[chain_id][res_num] = AMINO_ACIDS[res_name]

    # 组装每条链的序列
    sequences = {}
    for chain_id, residues in chains.items():
        # 按残基编号排序
        sorted_residues = [residues[k] for k in sorted(residues.keys())]
        sequences[chain_id] = ''.join(sorted_residues)

    return sequences


def extract_rna_sequence(pdb_file):
    """从核酸PDB文件中提取RNA/DNA序列"""
    if not os.path.exists(pdb_file):
        print(f"文件不存在: {pdb_file}")
        return {}

    # 存储序列，按链ID和残基编号组织
    chains = defaultdict(dict)

    # 跟踪已处理的残基，避免重复
    processed_residues = set()

    # PDB文件中ATOM记录的核苷酸残基
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            chain_id = line[21]
            res_num = int(line[22:26])
            res_name = line[17:20].strip()

            # 避免重复处理同一残基
            res_key = (chain_id, res_num)
            if res_key in processed_residues:
                continue

            # 检查是否是核苷酸
            base = None
            if res_name in NUCLEOTIDES:
                base = NUCLEOTIDES[res_name]
                processed_residues.add(res_key)
                chains[chain_id][res_num] = base

    # 组装每条链的序列
    sequences = {}
    for chain_id, residues in chains.items():
        # 按残基编号排序
        sorted_residues = [residues[k] for k in sorted(residues.keys())]
        sequences[chain_id] = ''.join(sorted_residues)
        # 如果序列中有T，很可能是DNA而不是RNA
        if 'T' in sequences[chain_id]:
            print(f"注意：链 {chain_id} 可能是DNA而不是RNA (包含T碱基)")

    return sequences


def save_sequence_to_fasta(pdb_id, sequences, output_file):
    """将单个PDB的序列保存为FASTA格式"""
    with open(output_file, 'w') as f:
        for chain_id, sequence in sequences.items():
            if sequence:  # 只保存非空序列
                f.write(f">{pdb_id}_{chain_id}\n")
                # 每行最多80个字符
                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i + 80]}\n")


def append_to_fasta(pdb_id, sequences, output_file):
    """将序列追加到FASTA文件"""
    # 如果文件不存在，创建它
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            pass  # 创建空文件

    # 追加序列
    with open(output_file, 'a') as f:
        for chain_id, sequence in sequences.items():
            if sequence:  # 只保存非空序列
                f.write(f">{pdb_id}_{chain_id}\n")
                # 每行最多80个字符
                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i + 80]}\n")


def process_all_pdbs(create_summary=True):
    """处理所有PDB文件并提取序列

    Args:
        create_summary: 是否创建汇总FASTA文件
    """
    # 创建输出目录
    os.makedirs(f"{OUTPUT_DIR}/sequences", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sequences/protein", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sequences/nucleic", exist_ok=True)

    # 获取所有蛋白质链和核酸链文件
    protein_files = glob.glob(f"{OUTPUT_DIR}/protein_chains/*.pdb")
    nucleic_files = glob.glob(f"{OUTPUT_DIR}/nucleic_chains/*.pdb")

    # 如果创建汇总文件，先清空
    if create_summary:
        summary_protein = f"{OUTPUT_DIR}/sequences/all_protein_sequences.fasta"
        summary_nucleic = f"{OUTPUT_DIR}/sequences/all_nucleic_sequences.fasta"
        # 清空或创建汇总文件
        open(summary_protein, 'w').close()
        open(summary_nucleic, 'w').close()

    protein_count = 0
    nucleic_count = 0

    # 处理蛋白质文件
    for pdb_file in protein_files:
        # 从文件名中提取PDB ID
        pdb_id = os.path.basename(pdb_file).split('_')[0]
        print(f"提取蛋白质序列: {pdb_id}")

        sequences = extract_protein_sequence(pdb_file)
        if sequences:
            protein_count += 1
            # 保存到单独文件
            output_file = f"{OUTPUT_DIR}/sequences/protein/{pdb_id}_protein.fasta"
            save_sequence_to_fasta(pdb_id, sequences, output_file)

            # 如果需要，添加到汇总文件
            if create_summary:
                append_to_fasta(pdb_id, sequences, summary_protein)

    # 处理核酸文件
    for pdb_file in nucleic_files:
        # 从文件名中提取PDB ID
        pdb_id = os.path.basename(pdb_file).split('_')[0]
        print(f"提取核酸序列: {pdb_id}")

        sequences = extract_rna_sequence(pdb_file)
        if sequences:
            nucleic_count += 1
            # 保存到单独文件
            output_file = f"{OUTPUT_DIR}/sequences/nucleic/{pdb_id}_nucleic.fasta"
            save_sequence_to_fasta(pdb_id, sequences, output_file)

            # 如果需要，添加到汇总文件
            if create_summary:
                append_to_fasta(pdb_id, sequences, summary_nucleic)

    print(f"提取了 {protein_count} 个蛋白质序列文件和 {nucleic_count} 个核酸序列文件。")
    print(f"单独序列文件已保存至：")
    print(f"- {OUTPUT_DIR}/sequences/protein/")
    print(f"- {OUTPUT_DIR}/sequences/nucleic/")

    if create_summary:
        print(f"汇总序列文件已保存至：")
        print(f"- {summary_protein}")
        print(f"- {summary_nucleic}")


def main():
    """主函数"""
    print("=== PDB文件处理和序列提取工具 ===")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查是否存在PDB文件
    pdb_files = glob.glob(f"{PDB_DIR}/*.pdb")
    if not pdb_files:
        print(f"指定目录 {PDB_DIR} 未找到PDB文件。请确保PDB文件位于正确目录。")
        return

    print(f"找到 {len(pdb_files)} 个PDB文件: {', '.join([os.path.basename(f) for f in pdb_files[:5]])}" +
          (f"..." if len(pdb_files) > 5 else ""))

    # 创建TCL脚本
    create_tcl_script()

    # 询问用户是否运行VMD
    choice = input("是否运行VMD分离链? (y/n): ").lower()
    vmd_success = False

    if choice == 'y':
        vmd_success = run_vmd_script()
    else:
        print("跳过VMD处理。")
        print(f"请确保{OUTPUT_DIR}/protein_chains和{OUTPUT_DIR}/nucleic_chains目录中已有分离的PDB文件。")

        # 检查目录是否存在
        if not os.path.exists(f"{OUTPUT_DIR}/protein_chains") or not os.path.exists(f"{OUTPUT_DIR}/nucleic_chains"):
            print("未找到必要的目录。请确保已分离蛋白质和核酸链。")
            return

    # 如果VMD处理成功或用户选择跳过，继续处理序列提取
    if vmd_success or choice != 'y':
        choice = input("是否提取序列? (y/n): ").lower()
        if choice == 'y':
            create_summary = input("是否创建汇总序列文件? (y/n): ").lower() == 'y'
            process_all_pdbs(create_summary)
        else:
            print("跳过序列提取。")


if __name__ == "__main__":
    main()