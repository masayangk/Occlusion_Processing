## 使い方

### 1. リポジトリのクローン (初回のみ)
次のコマンドを実行
```bash
cd
git clone http://tfsv.tasakilab:5051/git/kato-24/Occlusion_Processing.git
```

### 2. VSCodeをセットアップ
次の拡張機能をインストール
1. Docker
2. Dev Container

### 3. VSCodeでフォルダを開く
次のフォルダをVSCodeで開く
```bash
~/Occlusion_Processing
```

### 4. "コンテナで再度開く"をクリック
VSCodeの右下に表示される「コンテナで再度開く」をクリック


### Pipパッケージのインストール
```bash
pip install git+http://tfsv.tasakilab:5051/git/kato-24/Occlusion_Processing.git
```
または
```bash
cd /workspace
pip install .
```