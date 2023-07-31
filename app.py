from flask import Flask, render_template, request, redirect, url_for, flash, session
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta
from  flask_sqlalchemy import SQLAlchemy




top_gross = pd.read_csv('top_gross.csv')
movies = pd.read_csv('movies_data.csv')

tfidf_doc = TfidfVectorizer()
tfidf_doc_matrix = tfidf_doc.fit_transform(movies['document'])  
similarity_array = cosine_similarity(tfidf_doc_matrix)

cast_tfidf_object = TfidfVectorizer()
cast_tfidf_vector = cast_tfidf_object.fit_transform(movies['cast'])

director_tfidf_object = TfidfVectorizer()
director_tfidf_vector = director_tfidf_object.fit_transform(movies['director'])

genres_tfidf_object = TfidfVectorizer()
genres_tfidf_vector = genres_tfidf_object.fit_transform(movies['genres'])


db = SQLAlchemy()

app = Flask(__name__)
app.secret_key = "key1"
app.permanent_session_lifetime = timedelta(hours=1)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


# User model
class User(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    favourite_cast = db.Column(db.String(500))
    favourite_directors = db.Column(db.String(500))
    favourite_genres = db.Column(db.String(500))


    def __init__(self, username, password, favourite_cast, favourite_directors, favourite_genres):
        self.username = username
        self.password = password
        self.favourite_cast = favourite_cast
        self.favourite_directors = favourite_directors
        self.favourite_genres = favourite_genres

with app.app_context():
    db.create_all()


@app.route("/") 
def home():
    return render_template("index.html")


# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        
        user = User.query.filter_by(username=username).first()

        if user:
            flash('Username already exists!', 'error')
        else:
            new_user = User(username=username, password=password, favourite_cast='', favourite_directors='', favourite_genres='')
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. You can now login!', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session['logged_in'] = True
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again or Register with new UserName', 'error')
            return redirect(url_for('register'))
    else:
        if session.get('logged_in'):
            flash("Already logged in!")
            return redirect(url_for('dashboard'))

    return render_template('login.html')


# Dashboard route 
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please...login here')
        return redirect(url_for('login'))
    else:
        username = session['username']
        clean = lambda s : s.translate({ord(i): '' for i in ":-+;[].*#@$%^&()'\""})

        cast_ = User.query.filter_by(username=username).with_entities(User.favourite_cast).first()
        directors_ = User.query.filter_by(username=username).with_entities(User.favourite_directors).first()
        genres_ = User.query.filter_by(username=username).with_entities(User.favourite_genres).first()

    return render_template('dashboard.html', username=session['username'], cast=clean(str(cast_)).strip(','),
                                                                            director = clean(str(directors_)).strip(','),
                                                                            genres = clean(str(genres_).strip(',')).strip(','))                                 


# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


#User Profie Recommendation route
@app.route('/UserProfieRecommendation', methods=['POST'])
def UserProfieRecommendation():#favourite_cast = '', favourite_director = '', favourite_genres = ''
    favourite_cast = request.form['cast']
    favourite_directors = request.form['director']
    favourite_genres = request.form['genres']



    if 'username' in session:
        username = session['username']
    
        record = User.query.filter_by(username=username).first()
        record.favourite_cast = favourite_cast
        record.favourite_directors = favourite_directors
        record.favourite_genres = favourite_genres
        db.session.commit()

        cast_ = User.query.filter_by(username=username).with_entities(User.favourite_cast).first()
        directors_ = User.query.filter_by(username=username).with_entities(User.favourite_directors).first()
        genres_ = User.query.filter_by(username=username).with_entities(User.favourite_genres).first()

        clean = lambda s : s.translate({ord(i): '' for i in ":-+;[].*#@$%^&()'\""}) 

        cast = clean(str(cast_)).replace(' ', '').replace(',', ' ').lower()
        directors = clean(str(directors_)).replace(' ', '').replace(',', ' ').lower()
        genres = clean(str(genres_)).replace(' ', '').replace(',', ' ').lower()
                                 
        
        cast_vector, director_vector, genres_vector = (cast_tfidf_object.transform([cast]), director_tfidf_object.transform([directors]), 
                                                       genres_tfidf_object.transform([genres]))


        cast_similarities, director_similarities, genres_similarities = (cosine_similarity(cast_tfidf_vector, cast_vector), 
                                                                        cosine_similarity(director_tfidf_vector, director_vector), 
                                                                        cosine_similarity(genres_tfidf_vector, genres_vector))

        user_input_similarity_scores = (cast_similarities * 5) + (director_similarities) + (genres_similarities * 0.5)

        movies['user_recommendation_score'] = user_input_similarity_scores[:,0]

        recommendations = movies[movies['user_recommendation_score'] > 0.0] #

        if len(recommendations) == 0:
            flash("You haven't given any input!..or..Your input is not matching the content in my database")
            return redirect(url_for('dashboard'))
        
        else:
            if len(genres) < 5:
                flash("Sorted in ascending order based on the year")
                recommendations = recommendations.sort_values(by=['release_year'], ascending=True)

            else:
                recommendations = recommendations.sort_values(by=['user_recommendation_score'], ascending=False)#[:100]

            return render_template("profile_recommend.html", favourite_cast = clean(str(cast_)).strip(',').upper(),
                                                    favourite_director = clean(str(directors_)).strip(',').upper(),
                                                    favourite_genres = clean(str(genres_).upper()).strip(','),
                                                    data = recommendations.to_dict(orient='records'))
    
    else:
        flash("You haven't been logged out!..Please login here again")
        return redirect(url_for('login'))


# recommend ui /search route
@app.route("/recommend")
def recommend_UI():
    return render_template("recommend_UI.html")


#popular matches route
@app.route("/popular_matches", methods=['POST'])
def popular_matches():
    title = request.form.get("search-input")

    clean = lambda s : s.translate({ord(i): '' for i in " :-+;[].*#@$%^&()'\","}) 
    title_cleaned = clean(title.lower())  #removing special characters and changing into lower case

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



# recommend movies route
@app.route("/recommend_movies/<string:title>/<int:release_year>")
def recommend(title, release_year):
    
    idx =  movies.loc[(movies['title'] == title) & (movies['release_year'] == release_year)].index[0]
    score_lst = list(enumerate(similarity_array[idx]))   
    score_lst = sorted(score_lst, key = lambda x:x[1] , reverse=True)     
    score_lst = score_lst[0:25] 
    
    
    recommend_list = []                                             
    for i in range(len(score_lst)):
        a = score_lst[i][0]
        recommend_list.append(movies.iloc[a].to_dict())


    return render_template("recommend.html", data=recommend_list[1:], searched_movie=recommend_list[0])


# top gross movies list route
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
