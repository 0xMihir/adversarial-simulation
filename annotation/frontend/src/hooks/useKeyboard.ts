import { useEffect } from "react";

interface KeyboardHandlers {
  onConfirm?: () => void; // Enter
  onReject?: () => void; // Backspace
  onUndo?: () => void; // Z
  onNext?: () => void; // N
  onEscape?: () => void; // Escape
  onToggleLayer?: (n: number) => void; // 1-6
  onModeSelect?: () => void; // S
  onModeConnect?: () => void; // C
  onModeEdit?: () => void; // E

  // Added unified list navigation
  onArrowUp?: () => void; // ↑
  onArrowDown?: () => void; // ↓
}

export function useKeyboard(handlers: KeyboardHandlers) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Don't intercept when typing in an input/editable field
      const target = e.target as HTMLElement | null;
      const isTyping =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        !!target?.isContentEditable;

      if (isTyping) return;

      switch (e.key) {
        case "Escape":
          e.preventDefault();
          handlers.onEscape?.();
          break;
        case "Enter":
          e.preventDefault();
          handlers.onConfirm?.();
          break;
        case "Backspace":
          e.preventDefault();
          handlers.onReject?.();
          break;
        case "ArrowUp":
          e.preventDefault();
          handlers.onArrowUp?.();
          break;
        case "ArrowDown":
          e.preventDefault();
          handlers.onArrowDown?.();
          break;
        case "z":
        case "Z":
          if (!e.metaKey && !e.ctrlKey) handlers.onUndo?.();
          break;
        case "n":
        case "N":
          handlers.onNext?.();
          break;
        case "s":
        case "S":
          handlers.onModeSelect?.();
          break;
        case "c":
        case "C":
          if (!e.metaKey && !e.ctrlKey) handlers.onModeConnect?.();
          break;
        case "e":
        case "E":
          handlers.onModeEdit?.();
          break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
        case "6":
          if (!e.ctrlKey && !e.metaKey) handlers.onToggleLayer?.(parseInt(e.key) - 1);
          break;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}
