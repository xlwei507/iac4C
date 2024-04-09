import sys
from Bio import SeqIO

def extract_sequences(peak_file, fasta_file, output_file, bed_file):
    # Read the FASTA file
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    # DNA to mRNA conversion dictionary
    trans_dict = {'A': 'U', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    # Open your peak file and read each line
    with open(peak_file, "r") as file, open(output_file, "w") as out, open(bed_file, "w") as bed_out:
        for line in file:
            cols = line.strip().split("\t")
            chr = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            peak_id = cols[3]

            # Extract the corresponding sequence
            seq = genome[chr].seq[start:end]

            # Check if the sequence contains 'G'
            if 'G' in seq:
                # If it contains 'G', keep this 'G'
                extended_seq = genome[chr].seq[max(0, start-100):min(len(genome[chr].seq), end+100)]
                # Convert the DNA sequence to mRNA sequence
                mRNA_seq = ''.join([trans_dict.get(base, '') for base in extended_seq])
                out.write(f">{peak_id}\n{mRNA_seq}\n")
                
                # Output the 'G' position to the bed file
                g_positions = [i for i, base in enumerate(seq) if base == 'G']
                for pos in g_positions:
                    bed_out.write(f"{chr}\t{start + pos}\t{start + pos + 1}\t{peak_id}\n")

            else:
                # If it does not contain 'G', extract the nearest 'G'
                nearest_upstream_g = genome[chr].seq.rfind('G', 0, start) # Extract upstream
                nearest_downstream_g = genome[chr].seq.find('G', end) # Extract downstream
                if nearest_upstream_g != -1:
                    extended_seq = genome[chr].seq[max(0, nearest_upstream_g-100):min(len(genome[chr].seq), nearest_upstream_g+101)]
                    # Convert the DNA sequence to mRNA sequence
                    mRNA_seq = ''.join([trans_dict.get(base, '') for base in extended_seq])
                    out.write(f">{peak_id}_UP_nearest_G\n{mRNA_seq}\n")
                    bed_out.write(f"{chr}\t{nearest_upstream_g}\t{nearest_upstream_g + 1}\t{peak_id}_UP_nearest_G\n")
                if nearest_downstream_g != -1:
                    extended_seq = genome[chr].seq[max(0, nearest_downstream_g-100):min(len(genome[chr].seq), nearest_downstream_g+101)]
                    # Convert the DNA sequence to mRNA sequence
                    mRNA_seq = ''.join([trans_dict.get(base, '') for base in extended_seq])
                    out.write(f">{peak_id}_DOWN_nearest_G\n{mRNA_seq}\n")
                    bed_out.write(f"{chr}\t{nearest_downstream_g}\t{nearest_downstream_g + 1}\t{peak_id}_DOWN_nearest_G\n")

if __name__ == "__main__":
    peak_file = sys.argv[1]
    fasta_file = sys.argv[2]
    output_file = sys.argv[3]
    bed_file = sys.argv[4]
    extract_sequences(peak_file, fasta_file, output_file, bed_file)

