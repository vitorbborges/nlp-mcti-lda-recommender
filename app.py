import gradio as gr
import random
import pandas as pd

opo = pd.read_csv('oportunidades_results.csv', lineterminator='\n')
# opo = opo.iloc[np.where(opo['opo_brazil']=='Y')]
simulation = pd.read_csv('simulation2.csv')
userID = max(simulation['userID']) + 1

def build_display_text(opo_n):
    
    title = opo.loc[opo_n]['opo_titulo']
    link = opo.loc[opo_n]['link']
    summary = opo.loc[opo_n]['facebook-bart-large-cnn_results']

    display_text = f"**{title}**\n\nURL:\n{link}\n\nSUMMARY:\n{summary}"

    return display_text

opo_n_one = random.randrange(len(opo))
opo_n_two = random.randrange(len(opo))
opo_n_three = random.randrange(len(opo))
opo_n_four = random.randrange(len(opo))

evaluated = []

def predict_next(option, nota):
    global userID
    global opo_n_one
    global opo_n_two
    global opo_n_three
    global opo_n_four
    global evaluated
    global opo
    global simulation

    selected = [opo_n_one, opo_n_two, opo_n_three, opo_n_four][int(option)-1]

    simulation = simulation.append({'userID': userID, 'itemID': selected, 'rating': nota}, ignore_index=True)
    evaluated.append(selected)
    
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

    opo_n_one = items[est.index(sorted(est)[-1])]
    opo_n_two = items[est.index(sorted(est)[-2])]
    opo_n_three = items[est.index(sorted(est)[-3])]
    opo_n_four = items[est.index(sorted(est)[-4])]

    return build_display_text(opo_n_one), build_display_text(opo_n_two), build_display_text(opo_n_three), build_display_text(opo_n_four)


with gr.Blocks() as demo:
    with gr.Row():
        one_opo = gr.Textbox(build_display_text(opo_n_one), label='Oportunidade 1')
        two_opo = gr.Textbox(build_display_text(opo_n_two), label='Oportunidade 2')

    with gr.Row():
        three_opo = gr.Textbox(build_display_text(opo_n_three), label='Oportunidade 3')
        four_opo = gr.Textbox(build_display_text(opo_n_four), label='Oportunidade 4')

    with gr.Row():
        option = gr.Radio(['1', '2', '3', '4'], label='Opção', value = '1')

    with gr.Row():
        nota = gr.Slider(1,5,step=1,label="Nota 1")

    with gr.Row():
        confirm = gr.Button("Confirmar")

        confirm.click(fn=predict_next,
               inputs=[option, nota],
               outputs=[one_opo, two_opo, three_opo, four_opo])

if __name__ == "__main__":
    demo.launch() 