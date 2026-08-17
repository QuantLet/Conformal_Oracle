import urllib.request, re, json, time
from pathlib import Path
PAPERS={"rahimikia2511.18578":"2511.18578","goel2410.11773":"2410.11773",
        "schmitt2602.03903":"2602.03903","zhong2603.22569":"2603.22569"}
TERMS=["num_samples","top_k","top-k","top_p","top-p","temperature","nucleus",
       "sampling","samples drawn","Monte Carlo sample","n_samples","seed",
       "generation config","greedy","quantile head","predictive distribution"]
out={}
for name,aid in PAPERS.items():
    rec={"arxiv":aid,"fetched":False,"hits":{}}
    for url in (f"https://arxiv.org/html/{aid}v1",f"https://arxiv.org/abs/{aid}"):
        try:
            html=urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"cfp-audit/1.0 (mailto:danpele@ase.ro)"}),timeout=90).read().decode("utf8","replace")
            txt=re.sub(r"<[^>]+>"," ",html); txt=re.sub(r"\s+"," ",txt)
            rec["fetched"]=True; rec["source"]=url; rec["chars"]=len(txt)
            for t in TERMS:
                n=len(re.findall(re.escape(t),txt,re.I))
                if n: rec["hits"][t]=n
            ctx=[]
            for m in re.finditer(r"(num_samples|top[_-]k|top[_-]p|temperature|nucleus)",txt,re.I):
                ctx.append(txt[max(0,m.start()-120):m.start()+120])
            rec["contexts"]=ctx[:6]
            break
        except Exception as e:
            rec["error"]=f"{type(e).__name__}: {e}"
        time.sleep(2)
    out[name]=rec; print(name,"->",rec.get("source","FAILED"),len(rec.get("hits",{})),"terms")
Path("analysis/litsurvey/raw.json").write_text(json.dumps(out,indent=1))
