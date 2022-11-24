const CopyWebpackPlugin = require('copy-webpack-plugin');
const path = require('path');

/** @type {import('webpack').Configuration} */
module.exports = {
  entry: "./bootstrap.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "development",
  experiments: {
    asyncWebAssembly: true
  },
  plugins: [
    new CopyWebpackPlugin({
      patterns: ['index.html']
    })
  ],
};
