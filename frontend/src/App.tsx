import { useRoute } from "./hooks";
import { NewResearch, RunsPage } from "./pages/NewResearch";
import { RunPage } from "./pages/RunPage";
import { TopBar } from "./TopBar";

export function App() {
  const route = useRoute();
  return (
    <div className="app">
      <a className="sr-only" href="#content">
        Skip to content
      </a>
      <TopBar section={route.page === "new" ? "new" : "runs"} />
      <div id="content">
        {route.page === "new" ? <NewResearch /> : null}
        {route.page === "runs" ? <RunsPage /> : null}
        {route.page === "run" ? <RunPage key={route.id} id={route.id} view={route.view} /> : null}
      </div>
    </div>
  );
}
