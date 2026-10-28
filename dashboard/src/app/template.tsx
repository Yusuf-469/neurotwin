import type { ReactNode } from "react";

// A template (unlike the layout) remounts on every navigation, so its CSS entrance animation
// plays as a page transition. Opacity/translate only, with no lingering transform afterwards, so
// sticky and fixed children behave normally once it finishes.
export default function Template({ children }: { children: ReactNode }) {
  return <div className="page-enter">{children}</div>;
}
