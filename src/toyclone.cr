require "set"
require "./toyclone/config"
require "./toyclone/errors"
require "./toyclone/input"
require "./toyclone/sanitize"
require "./toyclone/kernel"
require "./toyclone/indexing"
require "./toyclone/result"
require "./toyclone/output"
require "./toyclone/cli"
require "./toyclone/run"

module Toyclone
  VERSION = {{ `shards version #{__DIR__}`.chomp.stringify }}
  PROGRAM = "toyclone"
  SOURCE  = "https://github.com/kojix2/toyclone"

  def self.main(args = ARGV)
    parser = CLI::Parser.new
    config = parser.parse(args)

    case config.action
    when Action::Help
      puts config.help_message
    when Action::Version
      puts "#{PROGRAM} #{VERSION}"
    when Action::Fit
      Run.execute(config)
    end
  rescue ex : CliError | OptionParser::Exception
    STDERR.puts("error: #{ex.message}")
    parser = CLI::Parser.new
    STDERR.puts(parser.help_message)
    exit 1
  rescue ex : KernelError
    STDERR.puts("error: #{ex.message}")
    exit 1
  end
end
