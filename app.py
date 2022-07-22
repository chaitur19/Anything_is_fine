import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Team Anything Is Fine")
st.sidebar.title("Some of the analysis of business products ")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("## Product Review Analysis")
st.sidebar.markdown("A making of sentiment analysis on different businesses products ")

data_path = ("Yelp.csv")

def load_data():
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    return data


data = load_data()
st.dataframe(data)

st.sidebar.subheader("Show random tweets")
random_tweet = st.sidebar.radio('Select the Sentiment',('positive','negative','neutral'))


st.subheader("Here are some example of tweets according to your choice!")
st.markdown("1." + data.query("sentiments == @random_tweet")[['text']].sample(n=1).iat[0,0])

st.sidebar.markdown("### Number of tweets")
select = st.sidebar.selectbox('Visualization Type',['Histogram','PieChart'])

sentiment_count = data['sentiments'].value_counts()
sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Tweets':sentiment_count.values})

if st.sidebar.checkbox('Show',False,key='0'):
    st.markdown("### No. of tweets by sentiments ")
    if select=='Histogram':
        fig = px.bar(sentiment_count,x='Sentiments',y='Tweets',color='Tweets',height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count,values='Tweets',names='Sentiments')
        st.plotly_chart(fig)



st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour of day",0,23)
modified_data = data[data['date'].dt.hour == hour]
if st.sidebar.checkbox('Show',False,key='1'):
    st.markdown('## Tweets location based on time of day')
    st.markdown('%i tweets in %i:00 and %i:00' %(len(modified_data), hour, (hour+1)%24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data",False):
        st.write(modified_data)

st.sidebar.subheader("Breakdown Sentiments by city")
choice = st.sidebar.multiselect("Pick City", tuple(pd.unique(data["city"])))
if st.sidebar.checkbox("Show", False, key="5"):
    if len(choice) > 0:
        chosen_data = data[data["city"].isin(choice)]
        fig = px.histogram(chosen_data, x="city", y="sentiments",
                                histfunc="count", color="sentiments",
                                facet_col="sentiments", labels={"sentiments": "sentiment"})
        st.plotly_chart(fig)

    # Word cloud

st.sidebar.subheader("Word Cloud")
word_sentiment = st.sidebar.radio("Which Sentiment to Display?", tuple(pd.unique(data["sentiments"])))
if st.sidebar.checkbox("Show", False, key="6"):
    st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
    df = data[data["sentiments"]==word_sentiment]
    words = " ".join(df["text"])
    #processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith() and word != "RT"])
    processed_words = " ".join([word for word in words.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

