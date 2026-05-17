- Make sure you have Node installed (using NVM).
- Install Python if you wish to run all the example code (using pyenv).
- Make sure your Ruby is up-to-date (3.4.4 as of this writing) using ```rbenv```
- Run ```bundle install```
- npm install -g vega-cli vega-lite
- npm install -g mermaid.cli
- npm install -g mermaid-filter
- To render presentation, start server with ```marp --watch assets/presentations/value-articulation-guide.md assets/presentations/value-articulation-guide-ppt.html```.
- To start server locally, run ```bundle exec jekyll serve```.
- To build, run ```bundle exec jekyll build```
- To regenerate resume, run ```pandoc avishek.net/assets/resume/avishek-sen-gupta-resume-2026-feb.md -s -o avishek.net/assets/resume/avishek-sen-gupta-resume-2026-feb.pdf```

To export a post to docx with Mermaid diagrams rendered (run from site root):

```bash
sed 's|](/assets/|](/path/to/avishek.net/assets/|g' _posts/<post>.md | \
  MERMAID_FILTER_PUPPETEER_CONFIG="$HOME/.puppeteer.json" \
  pandoc -o assets/<output>.docx -F ~/.nvm/versions/node/v26.1.0/bin/mermaid-filter
```

The `sed` step rewrites root-relative image paths to absolute so pandoc can find them.

`mermaid-filter` uses Puppeteer/Chrome to render diagrams. Create `~/.puppeteer.json` with your Chrome path:

```json
{"executablePath":"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
```

Then set in your shell profile:

```bash
export MERMAID_FILTER_PUPPETEER_CONFIG="$HOME/.puppeteer.json"
```

To run Excalidraw locally, run the following:

```
nvm install --lts
git clone https://github.com/excalidraw/excalidraw.git
yarn

docker build -t excalidraw/excalidraw .
docker run --rm -dit --name excalidraw -p 5000:80 excalidraw/excalidraw:latest
```

Full instructions [here](https://docs.excalidraw.com/docs/introduction/development)
