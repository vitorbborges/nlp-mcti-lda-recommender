import gradio as gr
import random
import pandas as pd

opo = pd.read_csv('oportunidades.csv')
simulation = pd.read_csv('simulation1.csv')
userID = max(simulation['userID']) + 1

opo_n = random.randrange(len(opo))
evaluated = [opo_n]

def predict_next(inp):
    global userID
    global opo_n
    global evaluated
    global opo
    global simulation

    simulation = simulation.append({'userID': userID, 'itemID': opo_n, 'rating': inp}, ignore_index=True)
    evaluated.append(opo_n)
    
    from surprise import Reader
    reader = Reader(rating_scale=(1, 5))

    from surprise import Dataset
    data = Dataset.load_from_df(simulation[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    from surprise import SVDpp
    svdpp = SVDpp()
    svdpp.fit(trainset)

    items = list()
    est = list()

    for i in range(len(opo)):
        if i not in evaluated:
            items.append(i)
            est.append(svdpp.predict(userID, i).est)
    
    opo_n = items[est.index(max(est))]
    return opo.loc[opo_n]['opo_texto']

with gr.Blocks() as demo:
    gr.Markdown("# MCTI Recommender System")
    
    current_opo = gr.Textbox(opo.loc[opo_n]['opo_texto'], label='Oportunidade')
    nota = gr.Slider(1,5,step=1,label="Nota")
    confirm = gr.Button("Confirmar")

    confirm.click(fn=predict_next,
               inputs=nota,
               outputs=current_opo)

demo.launch() 