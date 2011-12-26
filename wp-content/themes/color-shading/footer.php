			<div id="footer">
				<div class="gutter">
					
					<p class="left">
						&copy; <?php echo date("Y");?> <?php bloginfo('name'); ?>
					</p>
					
					<p class="credits">
						<?php if ( get_option( 'theme_credits', true ) ) : ?>
						Powered by <a href="<?php echo esc_url('http://www.onedesigns.com/wordpress-themes/color-shading'); ?>" title="Color Shading WordPress Theme">Color Shading Theme</a>
                        <?php endif; ?>
                        
						<?php if ( get_option( 'author_credits', false ) ) : ?>
						designed by <a href="<?php echo esc_url('http://www.ramblingwood.com/'); ?>">Ramblingwood</a>
						<?php endif; ?>
					</p>
					
				</div>
			</div>
			
		</div>
<?php wp_footer(); ?>
	</body>
</html>