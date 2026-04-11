"use client";

import type { DemoFaultMode, DraftStrategy, VerificationSensitivity } from "@/lib/types";

type Props = {
  draftModel: string;
  verificationModel: string;
  draftStrategy: DraftStrategy;
  verificationSensitivity: VerificationSensitivity;
  demoFaultMode: DemoFaultMode;
  demoFaultCount: number;
  onDraftModelChange: (value: string) => void;
  onVerificationModelChange: (value: string) => void;
  onDraftStrategyChange: (value: DraftStrategy) => void;
  onVerificationSensitivityChange: (value: VerificationSensitivity) => void;
  onDemoFaultModeChange: (value: DemoFaultMode) => void;
  onDemoFaultCountChange: (value: number) => void;
  onClearChat: () => void;
};

const models = ["qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct"];

export function ControlBar(props: Props) {
  return (
    <div className="sticky top-0 z-20 flex flex-wrap items-center gap-3 rounded-2xl border border-stone-200 bg-panel/90 px-4 py-3 shadow-panel backdrop-blur">
      <ControlSelect label="Draft model" value={props.draftModel} onChange={props.onDraftModelChange} options={models} />
      <ControlSelect
        label="Verification model"
        value={props.verificationModel}
        onChange={props.onVerificationModelChange}
        options={models}
      />
      <ControlSelect
        label="Draft strategy"
        value={props.draftStrategy}
        onChange={(value) => props.onDraftStrategyChange(value as DraftStrategy)}
        options={["conservative", "balanced", "demo"]}
      />
      <ControlSelect
        label="Verification sensitivity"
        value={props.verificationSensitivity}
        onChange={(value) => props.onVerificationSensitivityChange(value as VerificationSensitivity)}
        options={["conservative", "balanced", "demo"]}
      />
      <ControlSelect
        label="Verification demo"
        value={props.demoFaultMode}
        onChange={(value) => props.onDemoFaultModeChange(value as DemoFaultMode)}
        options={["off", "deterministic"]}
      />
      <ControlSelect
        label="Fault count"
        value={String(props.demoFaultCount)}
        onChange={(value) => props.onDemoFaultCountChange(Number(value))}
        options={["1", "2"]}
      />
      <div className="ml-auto flex items-end">
        <button
          type="button"
          className="rounded-xl border border-stone-200 bg-white px-4 py-2 text-sm font-medium text-stone-700 transition hover:bg-stone-50"
          onClick={props.onClearChat}
        >
          Очистить чат
        </button>
      </div>
    </div>
  );
}

function ControlSelect(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
}) {
  return (
    <label className="flex min-w-[180px] flex-col gap-1 text-sm">
      <span className="font-medium text-mutedink">{props.label}</span>
      <select
        className="rounded-xl border border-stone-200 bg-white px-3 py-2 text-sm outline-none ring-0 transition focus:border-stone-400"
        value={props.value}
        onChange={(event) => props.onChange(event.target.value)}
      >
        {props.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
