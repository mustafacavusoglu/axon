package http

import (
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/gofiber/fiber/v2/middleware/requestid"
)

func NewApp(handler *Handler) *fiber.App {
	app := fiber.New(fiber.Config{
		IdleTimeout:  120 * time.Second,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}
			return c.Status(code).JSON(fiber.Map{
				"error": err.Error(),
			})
		},
	})

	app.Use(requestid.New())
	app.Use(logger.New(logger.Config{
		Format:     "${time} [${pid}] ${locals:requestid} | ${status} | ${latency} | ${method} ${path}\n",
		TimeFormat: time.RFC3339,
	}))
	app.Use(recover.New())

	handler.RegisterRoutes(app)

	return app
}
