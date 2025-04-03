import { useEffect, useRef, useState } from "react"
import { cn } from "../../lib/utils"

export const PixelTrail = ({
  pixelSize = 16,
  delay = 0,
  fadeDuration = 1000,
  pixelClassName = "",
}: {
  pixelSize?: number
  delay?: number
  fadeDuration?: number
  pixelClassName?: string
}) => {
  const [pixels, setPixels] = useState<Array<{ x: number; y: number; id: number }>>(
    []
  )
  const containerRef = useRef<HTMLDivElement>(null)
  const mousePosition = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const requestRef = useRef<number>(0)
  const previousTimeRef = useRef<number>(0)
  const pixelId = useRef(0)

  useEffect(() => {
    const updateMousePosition = (e: MouseEvent) => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      mousePosition.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      }
    }

    containerRef.current?.addEventListener("mousemove", updateMousePosition)

    return () => {
      containerRef.current?.removeEventListener("mousemove", updateMousePosition)
    }
  }, [])

  const animate = (time: number) => {
    if (previousTimeRef.current !== undefined) {
      if (time - previousTimeRef.current >= delay) {
        setPixels((prevPixels) => [
          ...prevPixels.slice(-50),
          {
            x: mousePosition.current.x - pixelSize / 2,
            y: mousePosition.current.y - pixelSize / 2,
            id: pixelId.current,
          },
        ])
        pixelId.current += 1
        previousTimeRef.current = time
      }
    } else {
      previousTimeRef.current = time
    }
    requestRef.current = requestAnimationFrame(animate)
  }

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate)
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current)
      }
    }
  }, [delay])

  useEffect(() => {
    if (fadeDuration <= 0) return

    const interval = setInterval(() => {
      setPixels((prevPixels) => prevPixels.slice(-50))
    }, fadeDuration)

    return () => clearInterval(interval)
  }, [fadeDuration])

  return (
    <div ref={containerRef} className="relative w-full h-full">
      {pixels.map((pixel) => (
        <div
          key={pixel.id}
          className={cn(
            "absolute pointer-events-none",
            pixelClassName
          )}
          style={{
            left: pixel.x,
            top: pixel.y,
            width: pixelSize,
            height: pixelSize,
          }}
        />
      ))}
    </div>
  )
} 