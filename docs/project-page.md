# Publishing the project page

The existing repository can host a GitHub Pages project site at `https://sheepice.github.io/GeoBFM/`. Choose this option to keep the code and website under the current owner.

1. Create a `gh-pages` branch with the static site's `index.html` at its root. Put the page assets on the same branch and use relative links for styles, images and videos so they work under `/GeoBFM/`.
2. In the repository's **Settings > Pages**, select **Deploy from a branch**, choose **gh-pages** and **/(root)**, then save. Open the URL reported by Pages after deployment finishes.
3. Set the repository's **About > Website** field to the live page and add a project-page link to the README. On the site, link **Code** back to this repository. Once the paper has an arXiv identifier, add the real abstract URL to both the site and the README.

For an organization-style address such as `https://geobfm.github.io/`, the GitHub account or organization named `geobfm` must own the Pages repository `geobfm.github.io`. That site can link to `sheepice/GeoBFM` while the code stays here. A custom domain can also point to this repository's Pages site through its Pages settings and DNS.

[GitHub Pages site types](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) · [Configure a publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
