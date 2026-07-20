import clsx from "clsx";

const SIZES = {
  sm: "h-1 w-1",
  md: "h-1.5 w-1.5",
  lg: "h-2.5 w-2.5",
};

const GAPS = {
  sm: "gap-1",
  md: "gap-1.5",
  lg: "gap-2.5",
};

export default function LoadingDots({
  size = "md",
  className,
}: {
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <span className={clsx("inline-flex items-center", GAPS[size], className)}>
      <span
        className={clsx("animate-bounce rounded-full bg-current", SIZES[size])}
        style={{ animationDelay: "-0.3s" }}
      />
      <span
        className={clsx("animate-bounce rounded-full bg-current", SIZES[size])}
        style={{ animationDelay: "-0.15s" }}
      />
      <span className={clsx("animate-bounce rounded-full bg-current", SIZES[size])} />
    </span>
  );
}
