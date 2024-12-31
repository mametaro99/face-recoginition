from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, FieldList, FormField, HiddenField
from wtforms.validators import InputRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.file import FileField, FileAllowed
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

migrate = Migrate(app, db)  # Flask-Migrateの設定

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    face_image = db.Column(db.String(150), nullable=True)
    
    # 目線パターンを4つのカラムとして保存
    eye_pattern_1 = db.Column(db.String, nullable=True)
    eye_pattern_2 = db.Column(db.String, nullable=True)
    eye_pattern_3 = db.Column(db.String, nullable=True)
    eye_pattern_4 = db.Column(db.String, nullable=True)

class FaceRecognitionForm(FlaskForm):
    face_image = FileField('Face Image', validators=[
        FileAllowed(['jpg', 'png'], 'Images only!')
    ])
    
    # 目線パターンを4つ選択できるように変更
    eye_pattern_1 = SelectField('Eye Pattern 1', choices=[
        ('left', '左'),
        ('right', '右'),
        ('center', '真ん中'),
        ('blink', 'まばたき')
    ])
    
    eye_pattern_2 = SelectField('Eye Pattern 2', choices=[
        ('left', '左'),
        ('right', '右'),
        ('center', '真ん中'),
        ('blink', 'まばたき')
    ])
    
    eye_pattern_3 = SelectField('Eye Pattern 3', choices=[
        ('left', '左'),
        ('right', '右'),
        ('center', '真ん中'),
        ('blink', 'まばたき')
    ])
    
    eye_pattern_4 = SelectField('Eye Pattern 4', choices=[
        ('left', '左'),
        ('right', '右'),
        ('center', '真ん中'),
        ('blink', 'まばたき')
    ])


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('Remember me')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    confirm_password = PasswordField('Confirm Password', validators=[
        InputRequired(), EqualTo('password', message='Passwords must match')])

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)  # デフォルトのハッシュ方法を使用
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. You can now login.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    # 登録したユーザ情報を取得して表示
    user_data = {
        'username': current_user.username,
        'face_image': current_user.face_image,
        'eye_pattern_1': current_user.eye_pattern_1,
        'eye_pattern_2': current_user.eye_pattern_2,
        'eye_pattern_3': current_user.eye_pattern_3,
        'eye_pattern_4': current_user.eye_pattern_4,
    }
    return render_template('dashboard.html', user_data=user_data)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/face_recognition', methods=['GET', 'POST'])
@login_required
def face_recognition():
    form = FaceRecognitionForm()
    if form.validate_on_submit():
        # 保存先のディレクトリ
        upload_folder = os.path.join('static', 'uploads', 'faces')
        os.makedirs(upload_folder, exist_ok=True)

        # 顔画像を保存
        if form.face_image.data:
            filename = secure_filename(form.face_image.data.filename)
            filepath = os.path.join(upload_folder, filename)
            form.face_image.data.save(filepath)
            current_user.face_image = filepath

        # 目線パターンを4つのカラムに格納
        current_user.eye_pattern_1 = form.eye_pattern_1.data
        current_user.eye_pattern_2 = form.eye_pattern_2.data
        current_user.eye_pattern_3 = form.eye_pattern_3.data
        current_user.eye_pattern_4 = form.eye_pattern_4.data

        db.session.commit()
        flash('Face and eye patterns registered successfully.', 'success')
        return redirect(url_for('dashboard'))
    else:
        if request.method == 'POST':
            flash('Failed to register face and eye patterns. Please check the form.', 'error')
            # デバッグ用にエラーメッセージをログに出力
            for field, errors in form.errors.items():
                for error in errors:
                    app.logger.error(f"Error in {field}: {error}")

    return render_template('face_recognition.html', form=form)




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)