# STEP Viewer + SPH

ブラウザ完結・インストール不要のSTEPビューア。CADモデルを読み込み表示し、任意で水のSPHシミュレーションを重ねて動かせる(CPUバックエンド、STEP形状との衝突判定つき)。

## 構成

- `src/main.ts` — エントリーポイント。ファイル選択/D&D、透明度・再生コントロール(再生/ポーズ/停止/速度倍率)、カスタム水源フォームの配線
- `src/loader/` — STEP読み込み(occt-import-js、Web Worker実行)
- `src/scene/Viewer.ts` — Three.js WebGPURenderer(WebGL2へ自動フォールバック)。モデルは常にエッジ付き半透明表示(透明度は可変)、カメラ・グリッドの自動フィット、右下にCAD風のXYZ軸ギズモ
- `src/sim/` — SPH物理エンジン
  - `core/` — 物理モデル(パラメータ・カーネル関数・境界反射式)。CPU/GPU共通の仕様として設計
  - `geometry/` — STEP三角形メッシュ→Signed Distance Field変換(衝突判定用)
  - `backends/cpu/` — CPU実装(現在の唯一のバックエンド)
  - `SphSimulation.ts` — backend非依存の指揮者
- `scripts/copy-wasm.mjs` — `npm install`後に`occt-import-js.wasm`を`public/`へコピー(postinstall)

## セットアップ

```bash
npm install   # postinstallでoccct-import-js.wasmをpublic/へ自動コピー
npm run dev   # http://localhost:5173 で起動
```

`npm run build`で本番ビルド(`dist/`)を生成。静的ファイルなのでGitHub Pages等どこでもホスティング可能。

## GitHub Pagesへのデプロイ

`.github/workflows/deploy.yml`で、`main`ブランチへのpushをトリガーに自動ビルド&デプロイする設定済み。ユーザー側はNode.js/npm不要で、ブラウザでURLを開くだけで動く。

1. (初回のみ)GitHubリポジトリの Settings → Pages → Build and deployment → Source を **GitHub Actions** に設定
2. `main`ブランチにpushすると自動でビルド・デプロイされる
3. 公開URLは `https://<GitHubユーザー名>.github.io/<リポジトリ名>/`

`vite.config.ts`はGitHub Actions実行時に自動設定される`GITHUB_REPOSITORY`環境変数からbaseパス(`/<リポジトリ名>/`)を算出するため、リポジトリ名をハードコードする必要はない。ローカル開発時はbase`/`のまま影響なし。

## 使い方

1. STEPファイル(.step/.stp)をドラッグ&ドロップ、またはファイル選択で読み込み
2. 「モデル」パネルの透明度を調整(0=完全透明、1=不透明)
3. 「再生」で水シミュレーション開始(読み込んだ形状があればその上に、なければ箱の中に水塊が落下・衝突)。「ポーズ」でその場停止、「停止」で水を消去。速度倍率でスロー再生も可能
4. 「カスタム水源」パネルで、水源の中心位置・サイズ・粒子間隔(すべてmm)、発射方向・初速(m/s)を指定して「適用」

STEP座標はmm単位で来る前提で、読み込み時に×0.001してメートルに変換している(`STEP_UNIT_TO_METERS`, `src/main.ts`)。水源フォームの長さ系の値も同様にmm入力→内部でメートル変換している(`MM_TO_METERS`/`METERS_TO_MM`)。

## 既知の注意点

- `occt-import-js`のdist内`.wasm`パスは将来のバージョンで変わる可能性あり。`postinstall`が失敗した場合は`scripts/copy-wasm.mjs`のパスを確認
- `occt-import-js`はTypeScript型を同梱していないため、`src/loader/occt-import-js.d.ts`に最小限の型を手動定義している(使用しているAPIのみ)
- SDF(衝突判定用グリッド)の構築はメインスレッド同期実行。複雑な形状だと一瞬UIが固まる可能性がある
- SPHはCPUバックエンドのみ。WebGPU compute(TSL)への移植は`src/sim/core`の共有仕様を実装する形で将来対応予定
