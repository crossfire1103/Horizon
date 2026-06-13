"""CLI entry point for AI CTO Daily."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .storage.manager import ConfigError, StorageManager
from .orchestrator import AICTODailyOrchestrator
from .models import Config


console = Console()


def _config_with_topic(config: Config, topic_slug: str) -> Config:
    topic_slug = topic_slug.strip().lower()
    if topic_slug != "ai-cto" and not any(topic.slug == topic_slug for topic in config.topics):
        raise ValueError(f"Unknown topic: {topic_slug}")
    data = config.model_dump(mode="json")
    data["active_topic"] = topic_slug
    return Config.model_validate(data)


def print_banner():
    """Print the application banner."""
    banner = """
[bold blue]AI CTO Daily[/bold blue]
[cyan]  CTO-oriented AI technology briefing system[/cyan]
    """
    console.print(banner)


def main():
    """Main CLI entry point."""
    print_banner()

    parser = argparse.ArgumentParser(description="AI CTO Daily briefing pipeline")
    parser.add_argument("--hours", type=int, help="Force fetch from last N hours")
    parser.add_argument("--topic", help="Run a configured topic slug instead of active_topic")
    args = parser.parse_args()

    try:
        # Load environment variables from .env file
        load_dotenv()

        # Ensure we're in the project directory or use data/ in current dir
        data_dir = Path("data")

        # Initialize storage manager
        storage = StorageManager(data_dir=str(data_dir))

        # Load configuration
        try:
            config = storage.load_config()
        except FileNotFoundError:
            console.print("[bold red]❌ Configuration file not found![/bold red]\n")
            data_dir_path = data_dir if isinstance(data_dir, Path) else Path(data_dir)
            example_path = data_dir_path / "config.example.json"
            if example_path.exists():
                console.print(
                    f"Copy the example config and edit it:\n"
                    f"  [cyan]cp {example_path} {data_dir_path / 'config.json'}[/cyan]\n"
                )
            console.print(
                "Or run [bold cyan]uv run ai-cto-daily-wizard[/bold cyan] to launch the interactive setup wizard.\n"
            )
            sys.exit(1)
        except ConfigError as e:
            console.print(f"[bold red]❌ Error loading configuration: {e}[/bold red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]❌ Error loading configuration: {e}[/bold red]")
            sys.exit(1)

        if args.topic:
            try:
                config = _config_with_topic(config, args.topic)
            except ValueError as e:
                console.print(f"[bold red]{e}[/bold red]")
                sys.exit(1)

        # Create and run orchestrator
        orchestrator = AICTODailyOrchestrator(config, storage)
        asyncio.run(orchestrator.run(force_hours=args.hours))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_config_template():
    """Print configuration template."""
    template = """
{
  "version": "1.0",
  "ai": {
    "provider": "anthropic",
    "model": "claude-sonnet-4.5-20250929",
    "api_key_env": "ANTHROPIC_API_KEY",
    "temperature": 0.3,
    "max_tokens": 4096
  },
  "sources": {
    "github": [
      {
        "type": "user_events",
        "username": "torvalds",
        "enabled": true
      }
    ],
    "hackernews": {
      "enabled": true,
      "fetch_top_stories": 30,
      "min_score": 100
    },
    "rss": [
      {
        "name": "Example Blog",
        "url": "https://example.com/feed.xml",
        "enabled": true,
        "category": "software-engineering"
      }
    ]
  },
  "filtering": {
    "ai_score_threshold": 7.0,
    "time_window_hours": 24
  }
}

Also create a .env file with:
ANTHROPIC_API_KEY=your_api_key_here
GITHUB_TOKEN=your_github_token_here (optional but recommended)
"""
    console.print(template)


if __name__ == "__main__":
    main()
