from flask import Flask, render_template, jsonify, send_file, abort, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os, mimetypes

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///osman.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

PROJECTS_DIR = os.path.join(os.path.dirname(__file__), 'projects')

class Project(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), nullable=False)
    slug        = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    status      = db.Column(db.String(20), default='done')
    tech_stack  = db.Column(db.String(200))
    github_url  = db.Column(db.String(200))
    live_url    = db.Column(db.String(200))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    images      = db.Column(db.Text)

    def project_dir(self):
        return os.path.join(PROJECTS_DIR, self.slug)

    def get_file_tree(self):
        root = self.project_dir()
        if not os.path.isdir(root):
            return []
        tree = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == '.' else rel.count(os.sep) + 1
            if rel != '.':
                tree.append({
                    'type': 'dir',
                    'name': os.path.basename(dirpath),
                    'path': rel.replace('\\', '/'),
                    'depth': depth,
                })
            for fname in sorted(filenames):
                frel = os.path.join(rel, fname) if rel != '.' else fname
                tree.append({
                    'type': 'file',
                    'name': fname,
                    'path': frel.replace('\\', '/'),
                    'depth': depth,
                    'ext':  fname.rsplit('.', 1)[-1].lower() if '.' in fname else '',
                })
        return tree

    def get_screenshots(self):
        IMG_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}
        scr_dir = os.path.join(self.project_dir(), 'screenshots')
        if os.path.isdir(scr_dir):
            found = []
            for fname in sorted(os.listdir(scr_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMG_EXTS:
                    found.append({
                        'name': fname,
                        'url':  f'/project-image/{self.slug}/{fname}',
                    })
            if found:
                return found
        result = []
        for fname in [i.strip() for i in (self.images or '').split(',') if i.strip()]:
            result.append({
                'name': fname,
                'url':  f'/static/images/{fname}',
            })
        return result

    def get_readme(self):
        for name in ['README.md', 'readme.md', 'Readme.md']:
            p = os.path.join(self.project_dir(), name)
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
        return None

    def to_dict(self, full=False):
        d = {
            'id':          self.id,
            'name':        self.name,
            'slug':        self.slug,
            'description': self.description,
            'status':      self.status,
            'tech_stack':  [t.strip() for t in (self.tech_stack or '').split(',') if t.strip()],
            'github_url':  self.github_url,
            'live_url':    self.live_url,
            'created_at':  self.created_at.strftime('%Y-%m-%d'),
            'images':      self.get_screenshots(),
            'files':       self.get_file_tree(),
        }
        if full:
            d['readme'] = self.get_readme()
        return d

def seed():
    if Project.query.count() > 0:
        return
    projects = [
        #Project(name="Megga App", slug="megga", description="Megga Application", status="ongoing", tech_stack="Python, Flask, WebSocket, NLP", github_url="https://github.com/martenzo7/megga", images=""),
        Project(name="sentiment AI", slug="sentiment-ai", description="Bidirectional LSTM with Bahdanau Attention for real-time sentiment classification, served via a Flask web app with attention weight visualization.", status="ongoing", tech_stack="PyTorch, NLP, Flask, Deep Learning, LSTM", github_url="https://github.com/martenzo7", images=""),
    ]
    db.session.add_all(projects)
    db.session.commit()


@app.route('/')
@app.route('/projects')
@app.route('/projects/<path:slug>')
def spa(slug=None):
    return render_template('spa.html')

@app.route('/api/projects')
def api_projects():
    done    = [p.to_dict() for p in Project.query.filter_by(status='done').order_by(Project.created_at.desc()).all()]
    ongoing = [p.to_dict() for p in Project.query.filter_by(status='ongoing').order_by(Project.created_at.desc()).all()]
    return jsonify({'done': done, 'ongoing': ongoing})

@app.route('/api/projects/<slug>')
def api_project(slug):
    p = Project.query.filter_by(slug=slug).first_or_404()
    return jsonify(p.to_dict(full=True))

@app.route('/api/projects/<slug>/file/<path:filepath>')
def api_file(slug, filepath):
    """Serve raw file content from projects/<slug>/<filepath>"""
    p = Project.query.filter_by(slug=slug).first_or_404()
    safe_base = os.path.realpath(p.project_dir())
    full_path  = os.path.realpath(os.path.join(safe_base, filepath))
    if not full_path.startswith(safe_base):
        abort(403)
    if not os.path.isfile(full_path):
        abort(404)
    mime, _ = mimetypes.guess_type(full_path)
    if mime and not mime.startswith('text') and mime != 'application/json':
        abort(415)
    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    return jsonify({'content': content, 'path': filepath, 'name': os.path.basename(filepath)})

@app.route('/project-image/<slug>/<filename>')
def project_image(slug, filename):
    p = Project.query.filter_by(slug=slug).first_or_404()
    safe_base = os.path.realpath(os.path.join(p.project_dir(), 'screenshots'))
    full_path  = os.path.realpath(os.path.join(safe_base, filename))
    if not full_path.startswith(safe_base) or not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path)


@app.route('/download/cv')
def download_cv():
    cv_path = os.path.join(app.static_folder, 'files', 'osman_cv.pdf')
    if os.path.exists(cv_path):
        return send_file(cv_path, as_attachment=True, download_name='Osman_Abed_CV.pdf')
    abort(404)


with app.app_context():
    db.create_all()
    seed()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
