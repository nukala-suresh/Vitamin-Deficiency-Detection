from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
try:
    from vitamin_classifier import VitaminDeficiencyClassifier
except Exception as e:
    # If the ML dependencies (torch, timm, etc.) are not installed, importing
    # `vitamin_classifier` will fail. Catch that here and keep the app usable
    # for UI, auth and static pages. The classifier will be set to None later
    # if it cannot be instantiated.
    VitaminDeficiencyClassifier = None
    print(f"[WARN] Could not import vitamin_classifier: {e}. Model features disabled.")
import uuid
from dotenv import load_dotenv
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from datetime import datetime, timedelta
import secrets
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load Google AI API key from environment
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
app.config['SECRET_KEY'] = 'your-secret_key_here_12345'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
# If old DB exists in project root, move it to data/
old_db = os.path.join(os.path.dirname(__file__), 'users.db')
new_db = os.path.join(data_dir, 'users.db')
if os.path.exists(old_db) and not os.path.exists(new_db):
    try:
        os.replace(old_db, new_db)
    except Exception:
        pass
app.config['DATABASE'] = new_db
app.permanent_session_lifetime = timedelta(days=30)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize classifier with the new approach
try:
    classifier = VitaminDeficiencyClassifier(
        'vitamin_deficiency_model_weights.pth',  # weights file
        'class_info.json',
        'preprocessing_info.json',
        'model_config.json'  # new config file
    )
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    classifier = None


# --- Simple user database helpers (SQLite) ---
def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    conn = get_db_connection()
    # users table with optional email and verified flag
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            verified INTEGER DEFAULT 0
        )
    ''')
    # password reset tokens
    conn.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    ''')
    # email verification tokens
    conn.execute('''
        CREATE TABLE IF NOT EXISTS email_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    ''')
    # upload history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS upload_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            upload_date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

init_user_db()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user['id'], user['username'], user['email'])
    return None

# Expose a simple current_user in templates
@app.context_processor
def inject_user():
    return dict(current_user=current_user)



@app.route('/')
def index():
    # Always redirect to login page
    return redirect(url_for('login'))

@app.route('/home')
def home():
    """Public landing page"""
    return render_template('index.html')

@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()  # Optional
        password = request.form.get('password', '')
        if not username or not password:
            flash('Please provide both username and password')
            return redirect(url_for('register'))

        conn = get_db_connection()
        try:
            password_hash = generate_password_hash(password)
            # Email is optional, can be empty
            conn.execute('INSERT INTO users (username, email, password_hash, verified) VALUES (?, ?, ?, 1)', (username, email or None, password_hash))
            conn.commit()
            flash('Registration successful! Please login with your credentials.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken')
            return redirect(url_for('register'))
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'
        conn = get_db_connection()
        user_row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user_row and check_password_hash(user_row['password_hash'], password):
            user = User(user_row['id'], user_row['username'], user_row['email'])
            login_user(user, remember=remember)
            flash('Logged in successfully')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')


# --- Password reset flow ---
@app.route('/reset-request', methods=['GET', 'POST'])
def reset_request():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if not user:
            flash('If that account exists, a reset link has been sent.')
            conn.close()
            return redirect(url_for('login'))
        token = secrets.token_urlsafe(24)
        expires = (datetime.utcnow() + timedelta(hours=2)).isoformat()
        conn.execute('INSERT INTO password_resets (user_id, token, expires_at) VALUES (?, ?, ?)', (user['id'], token, expires))
        conn.commit()
        conn.close()
        send_password_reset(user['email'] or '', token)
        flash('If that account exists, a reset link has been sent.')
        return redirect(url_for('login'))
    return render_template('reset_request.html')


@app.route('/reset/<token>', methods=['GET', 'POST'])
def reset_password(token):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM password_resets WHERE token = ?', (token,)).fetchone()
    if not row:
        flash('Invalid or expired reset link')
        conn.close()
        return redirect(url_for('login'))
    expires = datetime.fromisoformat(row['expires_at'])
    if datetime.utcnow() > expires:
        flash('Reset link expired')
        conn.close()
        return redirect(url_for('login'))
    if request.method == 'POST':
        new_password = request.form.get('password', '')
        if not new_password:
            flash('Please provide a new password')
            return redirect(url_for('reset_password', token=token))
        password_hash = generate_password_hash(new_password)
        conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, row['user_id']))
        conn.execute('DELETE FROM password_resets WHERE id = ?', (row['id'],))
        conn.commit()
        conn.close()
        flash('Password updated. Please login.')
        return redirect(url_for('login'))
    conn.close()
    return render_template('reset_password.html', token=token)


def send_password_reset(email, token):
    reset_link = url_for('reset_password', token=token, _external=True)
    # Print reset link to console (no email sending)
    print(f"\n{'='*60}")
    print(f"🔑 PASSWORD RESET LINK (Console Output)")
    print(f"{'='*60}")
    print(f"Email: {email}")
    print(f"Link: {reset_link}")
    print(f"Valid for: 2 hours")
    print(f"{'='*60}\n")


# Admin CLI
@app.cli.command('list-users')
def list_users():
    conn = get_db_connection()
    rows = conn.execute('SELECT id, username, email FROM users').fetchall()
    conn.close()
    for r in rows:
        print(f"{r['id']}: {r['username']} <{r['email'] or 'No email'}>")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out')
    return redirect(url_for('index'))


@app.route('/account')
@login_required
def account():
    return render_template('account.html')


@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    # Get ONLY the current logged-in user's upload history
    user_id = current_user.id
    uploads = conn.execute('''
        SELECT * FROM upload_history 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (user_id,)).fetchall()
    conn.close()
    print(f"DEBUG: User {current_user.username} (ID: {user_id}) has {len(uploads)} uploads")
    return render_template('profile.html', uploads=uploads)


@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    """Clear upload history for the current logged-in user only"""
    conn = get_db_connection()
    user_id = current_user.id
    
    # Delete only THIS user's upload history
    conn.execute('DELETE FROM upload_history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    print(f"DEBUG: Cleared history for user {current_user.username} (ID: {user_id})")
    flash('Upload history cleared successfully')
    return redirect(url_for('profile'))


@app.route('/debug-db')
@login_required
def debug_db():
    """Debug route to check database contents"""
    conn = get_db_connection()
    all_uploads = conn.execute('SELECT * FROM upload_history').fetchall()
    users = conn.execute('SELECT id, username FROM users').fetchall()
    conn.close()
    
    output = "<h2>All Users:</h2>"
    for u in users:
        output += f"<p>ID: {u['id']}, Username: {u['username']}</p>"
    
    output += "<h2>All Upload History:</h2>"
    for upload in all_uploads:
        output += f"<p>ID: {upload['id']}, User_ID: {upload['user_id']}, Prediction: {upload['prediction']}, Date: {upload['upload_date']}</p>"
    
    output += f"<h2>Current User: {current_user.username} (ID: {current_user.id})</h2>"
    
    return output


@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    
    if not new_password or not confirm_password:
        flash('Please fill in all password fields')
        return redirect(url_for('account'))
    
    if new_password != confirm_password:
        flash('New passwords do not match')
        return redirect(url_for('account'))
    
    # Update password
    conn = get_db_connection()
    password_hash = generate_password_hash(new_password)
    conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, current_user.id))
    conn.commit()
    conn.close()
    
    flash('Password changed successfully')
    return redirect(url_for('account'))


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(url_for('upload_page'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('upload_page'))
    
    # Collect reported symptoms from the form
    reported_symptoms = request.form.getlist('symptoms')
    other_symptoms = request.form.get('other_symptoms', '').strip()
    if other_symptoms:
        reported_symptoms.append(other_symptoms)

    if file and classifier:
        try:
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction (pass reported symptoms for contextualization)
            result = classifier.predict(filepath, reported_symptoms=reported_symptoms)
            
            # Save to upload history
            conn = get_db_connection()
            user_id = current_user.id
            print(f"DEBUG: Saving upload for user {current_user.username} (ID: {user_id})")
            conn.execute('''
                INSERT INTO upload_history (user_id, filename, prediction, confidence, upload_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, filename, result['predicted_class'], result['confidence'], datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            # Return result page
            return render_template('result.html', result=result, filename=filename)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('upload_page'))
    else:
        flash('Error: Model not loaded properly')
        return redirect(url_for('upload_page'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)