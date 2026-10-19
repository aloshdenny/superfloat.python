# Developer docs (docs.superfloat.si)

The Mintlify project for the developer documentation. Hand-written MDX pages;
`docs.json` is the navigation. Edit the pages here and open a PR; every push to
`main` redeploys the site once this repository is connected in the Mintlify
dashboard with the docs directory set to `website`.

Preview locally: `npm i -g mint && cd website && mint dev` (http://localhost:3000).

Every Python output shown on a page was produced by running the code against
this library. When the library changes, re-run the examples and update the
pages in the same PR.
