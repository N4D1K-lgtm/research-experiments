import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Provider, cacheExchange, fetchExchange, Client } from "urql";
import App from "./App";

const client = new Client({
  url: "/graphql",
  exchanges: [cacheExchange, fetchExchange],
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Provider value={client}>
      <App />
    </Provider>
  </StrictMode>,
);
