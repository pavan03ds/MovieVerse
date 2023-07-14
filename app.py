from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def remove_special_characters(string):
    # Define the set of special characters to remove
    special_characters = " !@#$%^&*()_+=-{}[]|\:;\"'<>?,./"

    # Iterate over each character in the string and filter out the special characters
    cleaned_string = ''.join(char for char in string if char not in special_characters)

    return cleaned_string.lower()



#similarity_array = pickle.load(open('similarity_array.pkl','rb'))
top_gross = pd.read_csv('top_gross.csv')
movies = pd.read_csv('movies.csv')

tfidf_doc = TfidfVectorizer()
tfidf_doc_matrix = tfidf_doc.fit_transform(movies['document'])  

similarity_array = cosine_similarity(tfidf_doc_matrix)


app = Flask(__name__)

@app.route("/") 
def home():
    return render_template('index.html')

@app.route("/recommend")
def recommend_UI():
    return render_template("recommend_UI.html")


@app.route("/popular_matches", methods=['POST'])
def popular_matches():


    title = request.form.get("search-input")

    # clean = lambda s : s.translate({ord(i): '' for i in " :-,"}) 
    title_cleaned = remove_special_characters(title)  #removing special characters and changing into lower case


    if len(title_cleaned) == 0:
        statement_null = "Please... Enter a movie titile..."
        
        return render_template("popular_matches.html", statement=statement_null,
                                                        response = 404)
    
    else:

        popular_matches = movies.loc[movies['title_search']==title_cleaned]

        if len(popular_matches) == 0:
            statement = "The movie title you entered is not in our database... Please try entering a different title, or correct the spelling...."

            return render_template("popular_matches.html", statement=statement,
                                                                response = 404)
        else:
            return render_template("popular_matches.html", popular_matches = popular_matches.to_dict(orient='records'),
                                                                                                        response = 200)


@app.route("/recommend_movies/<string:title>/<int:release_year>")
def recommend(title, release_year):
    
    idx =  movies.loc[(movies['title'] == title) & (movies['release_year'] == release_year)].index[0]
    score_lst = list(enumerate(similarity_array[idx]))   #give index and score
    score_lst = sorted(score_lst, key = lambda x:x[1] ,reverse=True)     #sorting descending
    score_lst = score_lst[0:21] #excluding first item since it's the requested movie itself and giving top10
    
    
    recommend_list = []                                             
    for i in range(len(score_lst)):
        a = score_lst[i][0]
        recommend_list.append(movies.iloc[a].to_dict())  

    return render_template("recommend.html", data=recommend_list[1:], searched_movie=recommend_list[0])



@app.route("/topgross")
def topgross():
    return render_template('lists.html',
                           rank = list(top_gross['rank'].values),
                           title = list(top_gross['title_y'].values),
                           release_year=list(top_gross['year'].values),
                           poster=list(top_gross['poster'].values),
                           ww_gross=list(top_gross['worldwide_gross'].values)
                           )

if __name__=='__main__':
    app.run(debug=True)