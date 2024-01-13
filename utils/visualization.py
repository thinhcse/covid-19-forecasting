import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def infectious_plot(preds, truths, times, configs):
    
    predicts = [pred[0].asnumpy() for pred in preds]
    ground_truths = [truth[0].asnumpy() for truth in truths]

    if (len(times) != len(predicts)):
        predicts = predicts[:-1]
        ground_truths = ground_truths[:-1]

    df_preds = pd.DataFrame(data = predicts, columns = ['S', 'I', 'R', 'D'], index = times)
    df_truths = pd.DataFrame(data = ground_truths, columns= ['S', 'I', 'R', 'D'], index = times)

    df_preds = df_preds * 1E6
    df_truths = df_truths * 1E6

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df_preds.index, y = df_preds.I, name = 'infectious-predicts'))
    fig.add_trace(go.Scatter(x = df_truths.index, y = df_truths.I, name = 'infectious-truths'))
    fig.update_layout(title = f'Covid-19 in {configs["data"]["select_country"]} Prediction', font_family = 'Courier New')
    fig.show()

def recovered_plot(preds, truths, times, configs):
    
    predicts = [pred[0].asnumpy() for pred in preds]
    ground_truths = [truth[0].asnumpy() for truth in truths]

    if (len(times) != len(predicts)):
        predicts = predicts[:-1]
        ground_truths = ground_truths[:-1]

    df_preds = pd.DataFrame(data = predicts, columns = ['S', 'I', 'R', 'D'], index = times)
    df_truths = pd.DataFrame(data = ground_truths, columns= ['S', 'I', 'R', 'D'], index = times)

    df_preds = df_preds * 1E6
    df_truths = df_truths * 1E6

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df_preds.index, y = df_preds.R, name = 'recovered-predicts'))
    fig.add_trace(go.Scatter(x = df_truths.index, y = df_truths.R, name = 'recovered-truths'))
    fig.update_layout(title = f'Covid-19 in {configs["data"]["select_country"]} Prediction', font_family = 'Courier New')
    fig.show()

def deceased_plot(preds, truths, times, configs):
    
    predicts = [pred[0].asnumpy() for pred in preds]
    ground_truths = [truth[0].asnumpy() for truth in truths]

    if (len(times) != len(predicts)):
        predicts = predicts[:-1]
        ground_truths = ground_truths[:-1]

    df_preds = pd.DataFrame(data = predicts, columns = ['S', 'I', 'R', 'D'], index = times)
    df_truths = pd.DataFrame(data = ground_truths, columns= ['S', 'I', 'R', 'D'], index = times)

    df_preds = df_preds * 1E6
    df_truths = df_truths * 1E6

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df_preds.index, y = df_preds.D, name = 'deceased-predicts'))
    fig.add_trace(go.Scatter(x = df_truths.index, y = df_truths.D, name = 'deceased-truths'))
    fig.update_layout(title = f'Covid-19 in {configs["data"]["select_country"]} Prediction', font_family = 'Courier New')
    fig.show()