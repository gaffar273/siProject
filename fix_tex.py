# Fix the literal \n\n in the tex file
tex = open("nifty50_fed_rate_cuts_report.tex", "r", encoding="utf-8").read()
tex = tex.replace("behaviour:\\n\\n\\begin", "behaviour:\n\n\\begin")
open("nifty50_fed_rate_cuts_report.tex", "w", encoding="utf-8").write(tex)
print("Fixed! Lines:", tex.count("\n"))
