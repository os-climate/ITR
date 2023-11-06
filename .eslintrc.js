module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  extends: [
    // add generic rulesets here, such as:
    "eslint:recommended",
    "plugin:toml/standard",
  ],
  overrides: [
    {
      env: {
        node: true,
      },
      files: [".eslintrc.{js,cjs}"],
      parserOptions: {
        sourceType: "script",
      },
    },
  ],
  parser: "@babel/eslint-parser",
  parserOptions: {
    ecmaVersion: "latest",
  },
  overrides: [
    {
      files: ["*.toml"],
      parser: "toml-eslint-parser",
    },
  ],
  rules: {
    "comma-spacing": "error",
    "no-multi-spaces": ["error", { exceptions: { TOMLKeyValue: true } }],
    "no-multiple-empty-lines": "error",
    "no-trailing-spaces": "error",
  },
};
