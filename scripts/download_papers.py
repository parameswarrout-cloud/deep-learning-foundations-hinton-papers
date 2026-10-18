#!/usr/bin/env python3
"""
Paper Downloader for Geoffrey Hinton Papers Collection.
Downloads original research paper PDFs directly into the Papers/ directory.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

# Determine project root and destination folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "Papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Curated Roadmap papers and foundational classics
PAPER_DOWNLOADS = {
    # Phase 1: Core Engine
    "01_1986_Backpropagation.pdf": "https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf",
    "02_1986_Distributed_Representations.pdf": "https://www.cs.toronto.edu/~fritz/absps/pdp3.pdf",
    "03_2013_Initialization_and_Momentum.pdf": "https://www.cs.toronto.edu/~fritz/absps/momentum.pdf",
    
    # Phase 2: Architectural Blocks
    "04_2010_ReLU_Improve_RBMs.pdf": "https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf",
    "05_2014_Dropout.pdf": "https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf",
    "06_2016_Layer_Normalization.pdf": "https://arxiv.org/pdf/1607.06450.pdf",
    "07_1991_Adaptive_Mixtures_of_Local_Experts_MoE.pdf": "https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf",
    
    # Phase 3: Scaling & Practical Deep Learning
    "08_2012_AlexNet.pdf": "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf",
    "09_2008_t_SNE.pdf": "https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf",
    "10_2015_Distilling_the_Knowledge_in_a_Neural_Network.pdf": "https://arxiv.org/pdf/1503.02531.pdf",
    
    # Phase 4: Energy Models & Generative Thinking
    "11a_1985_Boltzmann_Machines.pdf": "https://www.cs.toronto.edu/~fritz/absps/cogscibm.pdf",
    "11b_2002_Contrastive_Divergence.pdf": "https://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf",
    "12_2006_Deep_Belief_Nets.pdf": "https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf",
    
    # Additional Historical & Foundational Classics
    "1994_Autoencoders_MDL_Helmholtz.pdf": "https://www.cs.toronto.edu/~fritz/absps/cvq.pdf",
    "1995_The_Helmholtz_Machine.pdf": "https://www.cs.toronto.edu/~fritz/absps/helmholtz.pdf",
    "1995_Wake_Sleep_Algorithm.pdf": "https://www.cs.toronto.edu/~hinton/absps/ws.pdf",
    "1999_Products_of_Experts.pdf": "https://www.cs.toronto.edu/~fritz/absps/icann-99.pdf",
    "2006_Reducing_Dimensionality_Science.pdf": "https://www.cs.toronto.edu/~hinton/absps/science.pdf",
    "2007_RBM_Collaborative_Filtering.pdf": "https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf",
    "2009_Deep_Boltzmann_Machines.pdf": "https://www.cs.toronto.edu/~fritz/absps/dbm.pdf",
    "2012_Acoustic_Modeling_Speech.pdf": "https://www.cs.toronto.edu/~hinton/absps/DNN-2012-proof.pdf",
    "2017_Dynamic_Routing_Between_Capsules.pdf": "https://www.cs.toronto.edu/~hinton/absps/DynamicRouting.pdf",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def download_all_papers(force_redownload: bool = False):
    print("=" * 60)
    print(f"Destination Directory: {PAPERS_DIR}")
    print(f"Total Papers in Catalog: {len(PAPER_DOWNLOADS)}")
    print("=" * 60)
    
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for filename, url in PAPER_DOWNLOADS.items():
        dest = PAPERS_DIR / filename
        
        # Check if already exists and non-empty (> 5 KB)
        if not force_redownload and dest.exists() and dest.stat().st_size > 5000:
            print(f"[SKIP - EXISTS] {filename} ({dest.stat().st_size // 1024} KB)")
            skipped_count += 1
            continue
        
        print(f"[DOWNLOADING] {filename} ...")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                
                # Check for small HTML error responses
                if len(content) < 5000:
                    print(f"  [WARNING] File too small ({len(content)} bytes), may be a block/captcha page.")
                    failed_count += 1
                    continue
                
                with open(dest, "wb") as f:
                    f.write(content)
                
                print(f"  [SAVED] {filename} ({len(content) // 1024} KB)")
                success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")
            failed_count += 1
        
        time.sleep(0.3)  # Polite crawling delay

    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count} downloaded, {skipped_count} skipped (already present), {failed_count} failed.")
    print("=" * 60)


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_all_papers(force_redownload=force)
