/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export: every page is client-rendered, so the dashboard can be served as plain files
  // by the FastAPI process (one container, one URL). trailingSlash emits page/index.html, which
  // Starlette's StaticFiles(html=True) serves for /page/.
  output: "export",
  trailingSlash: true,
};

export default nextConfig;
