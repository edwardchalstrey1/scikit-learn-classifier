import classifier
import statistics as st

repeats = 10
tt = []
pt = []
t = []

for i in range(0,repeats):
    
    report, results = classifier.results()
    tt.append(results["Training time (s)"])
    pt.append(results["Prediction time (s)"])
    t.append(results["Performance (micro avg f1 score)"])
    
results = {"Training time (s)": st.median(tt), "Prediction time (s)": st.median(pt),
    "Performance (micro avg f1 score)": st.median(t)}
print(results)
