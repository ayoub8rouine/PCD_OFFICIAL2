from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json

class ChatHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()
        self.wfile.write(b'')

    def do_GET(self):
        if self.path == '/':
            self._set_headers(200, 'text/plain')
            self.wfile.write(b'Server is running!')
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'Empty request'}).encode())
                return

            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
                return

            query = data.get('query', '').lower()
            response = self.generate_response(query)

            self._set_headers(200)
            self.wfile.write(json.dumps({'response': response}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())

    def generate_response(self, query):
        responses = {
            'headache': 'For headaches: rest, hydrate, use pain relievers, and cold compress.',
            'fever': 'For fever: hydrate, rest, use paracetamol or ibuprofen.',
            'cold': 'For colds: rest, fluids, decongestants, and honey for sore throat.',
        }
        for keyword, message in responses.items():
            if keyword in query:
                return message
        return "General advice: consult a healthcare professional for specific concerns."

def run(server_class=ThreadingHTTPServer, handler_class=ChatHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"🚀 Server running at http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
