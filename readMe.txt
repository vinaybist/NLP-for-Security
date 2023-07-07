2 things:
1. M.S. assignment Project
2. CVE to CAPEC mapping


Note:
Always check memory ==> RAM and storage (~10 GB)
$sudo lshw -c memory
$df -h

Some warnings if on low memory
2023-07-06 13:22:51.844322: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 27260928 exceeds 10% of free system memory.
os.environ["tf_gpu_allocator"]="cuda_malloc_async"


####Steps####
Git Clone
cd /NLP
Disable the local firewall (secure it via EC2 and IAM policies)  (check status via sudo ufw status)
sudo apt install python3-pip
sudo apt install python3.10-venv
Create a python env as cve -==> python3 -m venv cve
source cve/bin/activate
Run req.txt to nstall dependent packages as ==> $ bash req.txt

run as ==> $python app.py 
